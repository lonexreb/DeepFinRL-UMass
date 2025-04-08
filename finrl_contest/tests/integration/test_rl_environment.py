# tests/integration/test_rl_environment.py

import unittest
import numpy as np
import torch

from rl.environment.trading_env import TradingEnvironment
from rl.algorithms.ppo import PPO, PPOPolicy

class TestRLEnvironment(unittest.TestCase):
    def setUp(self):
        # Create a simple test configuration
        self.env_config = {
            'num_assets': 1,
            'observation_dim': 10,
            'initial_balance': 10000
        }
        
        # Sample data for testing
        self.test_data = {
            'prices': np.array([100, 105, 102, 110, 108, 112, 115, 113, 118, 120]),
            'fundamentals': np.array([90, 95, 92, 100, 98, 102, 105, 103, 108, 110])
        }
        
        # Create the environment
        self.env = TradingEnvironment(self.env_config, self.test_data)
    
    def test_environment_reset(self):
        """Test that the environment can be reset properly."""
        obs, _ = self.env.reset()
        
        # Check observation shape
        self.assertEqual(obs.shape, (self.env_config['observation_dim'],))
        
        # Check initial portfolio value
        self.assertEqual(self.env.portfolio_value, self.env_config['initial_balance'])
    
    def test_environment_step(self):
        """Test that the environment can take steps with actions."""
        self.env.reset()
        
        # Take a step with a random action
        action = np.array([1.0])  # Allocate 100% to the asset
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Check observation, reward, and done flag
        self.assertEqual(obs.shape, (self.env_config['observation_dim'],))
        self.assertIsInstance(reward, float)
        self.assertFalse(done)  # Should not be done after one step
        
        # Take multiple steps until done
        steps = 1
        while not done and steps < len(self.test_data['prices']) - 1:
            action = np.array([1.0])
            obs, reward, done, truncated, info = self.env.step(action)
            steps += 1
        
        # Should be done after all price data is used
        self.assertTrue(done or steps == len(self.test_data['prices']) - 1)
    
    def test_ppo_policy(self):
        """Test that the PPO policy can generate actions."""
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # Create policy
        policy = PPOPolicy(obs_dim, action_dim, hidden_dim=32)
        
        # Get observation
        obs, _ = self.env.reset()
        
        # Generate action
        action = policy.get_action(obs)
        
        # Check action shape and bounds
        self.assertEqual(action.shape, (action_dim,))
        self.assertTrue(np.all(action >= 0) and np.all(action <= 1))
        
        # Test deterministic action
        det_action = policy.get_action(obs, deterministic=True)
        self.assertEqual(det_action.shape, (action_dim,))
    
    def test_ppo_algorithm(self):
        """Test that the PPO algorithm can be initialized and collect rollouts."""
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # Create PPO agent
        rl_config = {
            'lr': 3e-4,
            'gamma': 0.99,
            'clip_ratio': 0.2,
            'hidden_dim': 32,
            'batch_size': 4  # Small batch size for testing
        }
        agent = PPO(obs_dim, action_dim, rl_config)
        
        # Collect rollouts
        num_steps = 5  # Collect a few steps for testing
        rollouts = agent.collect_rollouts(self.env, num_steps)
        
        # Check rollout data
        self.assertEqual(len(rollouts['states']), num_steps)
        self.assertEqual(len(rollouts['actions']), num_steps)
        self.assertEqual(len(rollouts['rewards']), num_steps)
        self.assertEqual(len(rollouts['returns']), num_steps)
        self.assertEqual(len(rollouts['advantages']), num_steps)
        
        # Try to update the policy
        loss = agent.update(rollouts)
        self.assertIsInstance(loss, float)

if __name__ == '__main__':
    unittest.main()
