import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from market_env import SynthMarketEnv

def evaluate_random_strategy(env):
    """A benchmark strategy that buys/sells/holds completely randomly."""
    obs, _ = env.reset()
    done = False
    net_worths = [env.balance]
    
    while not done:
        action = env.action_space.sample()  # Random action [0, 1, 2]
        obs, reward, done, truncated, info = env.step(action)
        
        current_net_worth = env.balance + (env.holdings * env.prices[env.current_step])
        net_worths.append(current_net_worth)
        
    return net_worths

def evaluate_rl_agent(env, model):
    """Evaluates our trained PPO agent."""
    obs, _ = env.reset()
    done = False
    net_worths = [env.balance]
    
    while not done:
        # Agent predicts the best action based on its training
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        current_net_worth = env.balance + (env.holdings * env.prices[env.current_step])
        net_worths.append(current_net_worth)
        
    return net_worths

def backtest():
    print("Loading the Closed-Loop Simulator...")
    env = SynthMarketEnv()
    
    print("Loading Trained RL Agent...")
    try:
        model = PPO.load("ppo_trading_agent")
    except FileNotFoundError:
        print("Error: 'ppo_trading_agent.zip' not found. Run train_agent.py first.")
        return
        
    # We want to test both strategies on the exact same unseen, hold-out synthetic episode.
    # We trigger a single reset (generating a novel market path), store it, and re-inject it.
    obs, _ = env.reset()
    holdout_prices = env.prices.copy()
    
    print("Backtesting RL Agent...")
    rl_net_worths = evaluate_rl_agent(env, model)
    
    print("Backtesting Random Benchmark...")
    # Reset env but forcefully inject the stored prices so the comparison is 1:1 fair
    env.reset()
    env.prices = holdout_prices
    random_net_worths = evaluate_random_strategy(env)
    
    # ------------------ Plotting the Results ------------------
    plt.figure(figsize=(14, 7))
    
    # RL Agent performance
    plt.plot(rl_net_worths, label='Our RL Agent (PPO)', color='green', linewidth=2.5)
    
    # Random Benchmark performance
    plt.plot(random_net_worths, label='Random Strategy', color='red', alpha=0.7, linestyle='--')
    
    # Market Price baseline (Buy & Hold simulation context)
    # We scale the price trajectory so it fits visually on the portfolio net worth graph
    initial_price = holdout_prices[env.window_size]
    market_performance = (holdout_prices[env.window_size:env.window_size+len(rl_net_worths)] / initial_price) * env.initial_balance
    plt.plot(market_performance, label='Market Price (Underlying Asset)', color='blue', alpha=0.3, linewidth=1)
    
    plt.title("Hold-Out Backtest: RL Agent vs Random on Unseen Synthetic Market Dynamics", fontsize=14)
    plt.xlabel("Trading Days", fontsize=12)
    plt.ylabel("Portfolio Net Worth ($)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("backtest_results.png")
    print("\nSUCCESS: Plotted backtest results to 'backtest_results.png'!")
    print("If the green line is consistently above the red line, your agent has learned to exploit market patterns.")
    plt.show()

if __name__ == "__main__":
    backtest()
