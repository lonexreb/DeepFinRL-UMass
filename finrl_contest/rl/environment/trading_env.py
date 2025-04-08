# rl/environment/trading_env.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TradingEnvironment(gym.Env):
    """A simplified trading environment for reinforcement learning."""
    
    def __init__(self, config, data):
        super(TradingEnvironment, self).__init__()
        
        self.config = config
        self.data = data
        self.current_step = 0
        self.max_steps = len(data['prices']) - 1
        
        # Define action and observation space
        # Action space: allocation percentage (0-1) for each asset
        self.num_assets = config.get('num_assets', 1)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.num_assets,), dtype=np.float32
        )
        
        # Observation space: market state (prices, fundamentals, etc.)
        # For simplicity, we'll use a fixed-size vector
        obs_dim = config.get('observation_dim', 10)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize portfolio
        self.initial_balance = config.get('initial_balance', 10000)
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.num_assets)
        self.portfolio_value = self.initial_balance
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.num_assets)
        self.portfolio_value = self.initial_balance
        
        return self._get_observation(), {}
    
    def step(self, action):
        # Ensure action is valid (sums to 1)
        action = np.clip(action, 0, 1)
        action = action / np.sum(action) if np.sum(action) > 0 else action
        
        # Calculate returns based on price changes
        old_prices = self.data['prices'][self.current_step]
        self.current_step += 1
        new_prices = self.data['prices'][self.current_step]
        
        # For simplicity, we'll use a single asset price
        price_change_pct = (new_prices - old_prices) / old_prices
        
        # Calculate portfolio return
        portfolio_return = np.sum(action * price_change_pct)
        self.portfolio_value *= (1 + portfolio_return)
        
        # Calculate reward (can be customized based on risk preferences)
        reward = portfolio_return
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Get new observation
        observation = self._get_observation()
        
        return observation, reward, done, False, {}
    
    def _get_observation(self):
        """Construct the observation from current state."""
        # For simplicity, we'll use a fixed-size vector with price and fundamental data
        obs_dim = self.observation_space.shape[0]
        observation = np.zeros(obs_dim)
        
        # Fill with available data (prices, fundamentals, etc.)
        # This is a simplified example - in practice, you'd include more features
        idx = min(self.current_step, len(self.data['prices']) - 1)
        observation[0] = self.data['prices'][idx]
        
        if 'fundamentals' in self.data and len(self.data['fundamentals']) > 0:
            fund_idx = min(idx, len(self.data['fundamentals']) - 1)
            observation[1] = self.data['fundamentals'][fund_idx]
        
        # Add portfolio state
        observation[2] = self.portfolio_value / self.initial_balance
        
        return observation
    
    def render(self):
        """Render the environment."""
        print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}")
