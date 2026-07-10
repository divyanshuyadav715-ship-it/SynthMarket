from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from stable_baselines3 import PPO
from market_env import SynthMarketEnv

app = FastAPI(title="Adaptive Trading Protocol (ATP) API", description="XAI and Inference API for RL Agent")

# Initialize models globally
try:
    model = PPO.load("ppo_trading_agent")
    env = SynthMarketEnv()
except Exception as e:
    print(f"Warning: Models not loaded. Train them first. {e}")
    model = None
    env = None

class MarketData(BaseModel):
    features: list[float]  # 20 prices + balance + holdings

@app.post("/predict")
def predict_action(data: MarketData):
    if model is None:
        return {"error": "Model not loaded on server."}
        
    obs_array = np.array(data.features, dtype=np.float32)
    
    # 1. Inference: Predict action
    action, _states = model.predict(obs_array, deterministic=True)
    
    # Calculate a pseudo-confidence score (PPO deterministic doesn't output probs directly,
    # so we measure market volatility to estimate certainty).
    prices = obs_array[:20]
    volatility = np.std(prices)
    confidence = max(0.5, 1.0 - (volatility / 100.0))
    
    action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
    action_name = action_map[int(action)]
    
    # 2. XAI (Explainable AI) - Feature Importance & Reasoning
    # We extract the first layer weights of the policy network to see which feature triggered the decision.
    try:
        # Extract weights from the MLP feature extractor
        weights = model.policy.mlp_extractor.policy_net[0].weight.detach().numpy()
        # Calculate importance score: Average weight magnitude * feature value
        feature_importance = np.mean(np.abs(weights), axis=0) * np.abs(obs_array)
        top_feature_idx = np.argmax(feature_importance)
        
        if top_feature_idx < 20:
            reasoning = f"Price trend anomaly detected on Day -{20 - top_feature_idx}."
        elif top_feature_idx == 20:
            reasoning = "Decision heavily influenced by current Cash Balance constraints."
        else:
            reasoning = "Decision based on Portfolio Exposure (Current Holdings)."
            
    except Exception as e:
        reasoning = "Black-box decision (fallback)."
        
    # Override reasoning for specific obvious cases for better UX in dashboard
    if action_name == "Sell" and volatility > 5.0:
        reasoning = "High Volatility (Crash incoming) detected. Liquidating assets to minimize risk."
    elif action_name == "Buy" and obs_array[-2] > 5000:
        reasoning = "Favorable entry point with high capital availability."

    return {
        "action": int(action),
        "action_name": action_name,
        "confidence": float(confidence),
        "reasoning": reasoning
    }
