#!/usr/bin/env python
# scripts/train.py

import argparse
import yaml
import os
import numpy as np
import torch
from pathlib import Path

# Import RL components
from rl.environment.trading_env import TradingEnvironment
from rl.algorithms.ppo import PPO

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def generate_sample_data(num_days=100):
    """Generate sample price and fundamental data for training."""
    # Generate random price series with some trend and volatility
    prices = [100]  # Start with price of 100
    for _ in range(num_days - 1):
        # Random price change with some momentum and mean reversion
        change = np.random.normal(0, 1) + 0.001 * (100 - prices[-1])
        new_price = max(prices[-1] * (1 + change * 0.01), 0.1)  # Ensure price is positive
        prices.append(new_price)
    
    # Generate fundamental data (e.g., earnings, book value)
    fundamentals = []
    for price in prices:
        # Fundamental value is related to price but with some noise
        fundamental = price * (0.8 + 0.4 * np.random.random())
        fundamentals.append(fundamental)
    
    return {
        'prices': np.array(prices),
        'fundamentals': np.array(fundamentals)
    }

def train(config, output_dir):
    """Train the RL agent."""
    # Generate sample data (in a real scenario, this would be loaded from a database or API)
    data = generate_sample_data(num_days=config.get('training', {}).get('num_days', 100))
    
    # Create the environment
    env_config = config.get('environment', {})
    env = TradingEnvironment(env_config, data)
    
    # Create the RL agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    rl_config = config.get('rl', {})
    agent = PPO(obs_dim, action_dim, rl_config)
    
    # Training parameters
    num_epochs = config.get('training', {}).get('num_epochs', 50)
    steps_per_epoch = config.get('training', {}).get('steps_per_epoch', 1000)
    
    # Training loop
    for epoch in range(num_epochs):
        # Collect experience
        rollouts = agent.collect_rollouts(env, steps_per_epoch)
        
        # Update policy
        loss = agent.update(rollouts)
        
        # Evaluate policy
        returns = rollouts['returns']
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f} | Mean Return: {mean_return:.4f} | Std Return: {std_return:.4f}")
        
        # Save model periodically
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            model_path = os.path.join(output_dir, f"ppo_model_epoch_{epoch+1}.pt")
            agent.save(model_path)
            print(f"Model saved to {model_path}")
    
    return agent

def main():
    parser = argparse.ArgumentParser(description='Train RL agent for trading')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='output/models',
                        help='Directory to save trained models')
    args = parser.parse_args()
    
    # Load configuration
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / args.config
    config = load_config(config_path)
    
    # Create output directory if it doesn't exist
    output_dir = base_dir / args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Add RL-specific configuration if not present
    if 'rl' not in config:
        config['rl'] = {
            'lr': 3e-4,
            'gamma': 0.99,
            'clip_ratio': 0.2,
            'hidden_dim': 64,
            'batch_size': 64
        }
    
    if 'environment' not in config:
        config['environment'] = {
            'num_assets': 1,
            'observation_dim': 10,
            'initial_balance': 10000
        }
    
    if 'training' not in config:
        config['training'] = {
            'num_epochs': 50,
            'steps_per_epoch': 1000,
            'num_days': 100
        }
    
    # Train the agent
    agent = train(config, output_dir)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
