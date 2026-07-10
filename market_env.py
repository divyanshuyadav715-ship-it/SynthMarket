import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from models import Generator

class SynthMarketEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    Loads a GAN Generator to create novel market episodes (including crashes and bull-runs)
    every time reset() is called. This provides an infinite stream of diverse data.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, seq_len=24, window_size=20, initial_balance=10000.0):
        super(SynthMarketEnv, self).__init__()
        self.seq_len = seq_len
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost_pct = 0.001  # 0.1% slippage/fees for realism
        
        # Load the pre-trained GAN Generator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(z_dim=10, hidden_dim=64, out_dim=1).to(self.device)
        try:
            self.generator.load_state_dict(torch.load("generator_model.pth", map_location=self.device))
        except Exception as e:
            print(f"Warning: Could not load 'generator_model.pth'. Make sure to train the GAN first. {e}")
            
        self.generator.eval()

        # Action space: 3 discrete actions [0: Hold, 1: Buy, 2: Sell]
        self.action_space = spaces.Discrete(3)
        
        # Observation space: Last 20 days of prices + current balance + current holdings (22 features total)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size + 2,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Step 1: Generate completely new synthetic market data
        with torch.no_grad():
            # Generate 50 sequences of 24 days to form a 1200 day episode
            z = torch.randn(50, self.seq_len, 10).to(self.device)
            fake_returns = self.generator(z).cpu().numpy().squeeze().flatten()
            
        # Convert synthetic scaled returns into a simulated price path.
        # Assuming base price is $100 and a generic volatility scalar.
        volatility_scale = 0.05
        prices = [100.0]
        for ret in fake_returns:
            # We construct a price based on compounding the generated synthetic return
            prices.append(prices[-1] * (1.0 + (ret * volatility_scale)))
            
        self.prices = np.array(prices, dtype=np.float32)
        
        # Step 2: Reset agent state
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.holdings = 0.0
        self.net_worth = self.initial_balance
        
        return self._next_observation(), {}

    def _next_observation(self):
        # Extract the trailing window of prices
        window = self.prices[self.current_step - self.window_size : self.current_step]
        # Append internal agent state variables
        obs = np.append(window, [self.balance, self.holdings])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        current_price = self.prices[self.current_step]
        prev_net_worth = self.balance + (self.holdings * current_price)
        
        # Execute trade
        if action == 1: # Buy 1 share
            if self.balance >= current_price:
                self.holdings += 1
                cost = current_price * (1 + self.transaction_cost_pct)
                self.balance -= cost
                
        elif action == 2: # Sell 1 share
            if self.holdings > 0:
                self.holdings -= 1
                revenue = current_price * (1 - self.transaction_cost_pct)
                self.balance += revenue
                
        elif action == 0: # Hold
            pass
            
        # Increment time
        self.current_step += 1
        
        # Calculate new net worth based on next step's market price
        new_price = self.prices[self.current_step] if self.current_step < len(self.prices) else current_price
        current_net_worth = self.balance + (self.holdings * new_price)
        self.net_worth = current_net_worth
        
        # Reward is the change in portfolio value (realized + unrealized PnL)
        reward = float(current_net_worth - prev_net_worth)
        
        # Stop episode if we run out of generated market data
        done = self.current_step >= len(self.prices) - 1
        truncated = False
        
        return self._next_observation(), reward, done, truncated, {}
