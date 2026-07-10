# 📈 SynthMarket: Autonomous RL Trading System with GANs & MLOps

![SynthMarket](https://img.shields.io/badge/Status-Production_Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C)
![Reinforcement Learning](https://img.shields.io/badge/RL-Stable_Baselines3-purple)

**SynthMarket** is an industry-grade algorithmic trading platform that combines Generative Adversarial Networks (GANs), Deep Reinforcement Learning (RL), and Explainable AI (XAI) with real-time Model Drift Monitoring.

Most quantitative trading models fail because they overfit to historical data. SynthMarket solves this by using a **Closed-Loop Simulator**. A Time-Series GAN generates infinite synthetic market scenarios (including black-swan events and crashes), and a Proximal Policy Optimization (PPO) RL Agent learns to navigate these scenarios without ever seeing real-world data during training. This makes the model incredibly robust against unexpected market volatility.

## 🌟 Key Features

1. **Market Simulator (Time-Series GAN)**
   - Custom LSTM-based Generator and Bidirectional LSTM Discriminator.
   - Learns temporal dynamics of real stock data (`yfinance`) and generates highly realistic synthetic price trajectories.
2. **Autonomous Trading Agent (Reinforcement Learning)**
   - Custom `Gymnasium` environment simulating slippage, transaction costs, and portfolio net worth.
   - PPO (Proximal Policy Optimization) agent trained purely on synthetic data to "sell before the crash".
3. **Live Ops & Drift Monitoring (MLOps)**
   - Real-time `Streamlit` dashboard for system health observability.
   - **KL Divergence Monitoring:** Automatically detects "Model Drift" by comparing the live market distribution against the GAN's baseline training distribution.
4. **Explainable AI (XAI)**
   - `FastAPI` backend runs real-time inference and performs feature importance extraction to explain *why* the agent decided to Buy, Sell, or Hold.

## 🏗️ Architecture

- `data_loader.py`: Ingests and normalizes historical stock data.
- `models.py`: PyTorch GAN architecture (Generator & Discriminator).
- `train.py`: Minimax training loop with gradient clipping.
- `visualize.py`: Validates synthetic data using Probability Density Function (PDF) overlap.
- `market_env.py`: Custom OpenAI Gym trading environment.
- `train_agent.py`: Trains the PPO Reinforcement Learning model.
- `backtest.py`: Evaluates the agent on a hold-out test set against a random benchmark.
- `api.py`: FastAPI inference and XAI endpoint.
- `monitor.py`: KL-Divergence statistical drift monitor.
- `dashboard.py`: Streamlit frontend for live visualization.

## 🚀 Getting Started

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Market Simulator (GAN)
```bash
python train.py
python visualize.py  # View the PDF validation plot
```

### 3. Train the Trading Agent (RL)
```bash
python train_agent.py
python backtest.py  # View the strategy performance vs random
```

### 4. Run the Live Ops Dashboard
Terminal 1 (Backend):
```bash
uvicorn api:app --reload
```
Terminal 2 (Frontend):
```bash
streamlit run dashboard.py
```

## 🧠 The "God-Mode" Pitch for Quant Roles
*"AI models in production degrade over time—a phenomenon known as Model Drift. I built an Ops-Layer that tracks Model Drift using KL Divergence. As soon as the real-time market behavior misaligns with the training data distribution, my monitor triggers a 'Drift Alert' and auto-pauses the bot to prevent catastrophic losses."*

---
*Disclaimer: This is a sophisticated simulation project intended for educational and portfolio purposes. Do not use for actual financial trading without rigorous live-market testing.*
