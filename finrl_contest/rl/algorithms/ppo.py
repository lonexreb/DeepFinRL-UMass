# rl/algorithms/ppo.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class PPOPolicy(nn.Module):
    """Actor-Critic policy network for PPO algorithm."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(PPOPolicy, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Policy head (actor)
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value head (critic)
        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        action_mean = self.policy_mean(features)
        action_std = torch.exp(self.policy_log_std).expand_as(action_mean)
        value = self.value(features)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_mean, action_std, _ = self.forward(state)
            
            if deterministic:
                action = action_mean
            else:
                normal = Normal(action_mean, action_std)
                action = normal.sample()
            
            # Apply sigmoid to ensure action is in [0, 1]
            action = torch.sigmoid(action)
            
        # Ensure the action has the correct shape (action_dim,)
        action_np = action.squeeze().numpy()
        # If action is a scalar, reshape it to a 1D array
        if np.isscalar(action_np) or action_np.ndim == 0:
            action_np = np.array([action_np])
            
        return action_np
    
    def evaluate_actions(self, states, actions):
        action_mean, action_std, value = self.forward(states)
        
        normal = Normal(action_mean, action_std)
        log_probs = normal.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        
        return log_probs, entropy, value

class PPO:
    """Proximal Policy Optimization algorithm."""
    
    def __init__(self, obs_dim, action_dim, config):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # Hyperparameters
        self.lr = config.get('lr', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.ppo_epochs = config.get('ppo_epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        
        # Initialize policy network
        self.policy = PPOPolicy(obs_dim, action_dim, config.get('hidden_dim', 64))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
    
    def update(self, rollouts):
        # Extract rollout data
        states = torch.FloatTensor(rollouts['states'])
        actions = torch.FloatTensor(rollouts['actions'])
        old_log_probs = torch.FloatTensor(rollouts['log_probs']).view(-1, 1)
        returns = torch.FloatTensor(rollouts['returns']).view(-1, 1)
        advantages = torch.FloatTensor(rollouts['advantages']).view(-1, 1)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update loop
        total_loss = 0
        for _ in range(self.ppo_epochs):
            # Create mini-batches
            indices = np.random.permutation(states.size(0))
            for start in range(0, states.size(0), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                # Get mini-batch data
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                
                # Evaluate actions
                new_log_probs, entropy, values = self.policy.evaluate_actions(mb_states, mb_actions)
                
                # Compute policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = nn.MSELoss()(values, mb_returns)
                
                # Compute entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        return total_loss / (self.ppo_epochs * (states.size(0) // self.batch_size))
    
    def collect_rollouts(self, env, num_steps):
        """Collect experience using the current policy."""
        rollouts = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        state, _ = env.reset()
        done = False
        
        for _ in range(num_steps):
            # Get action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_mean, action_std, value = self.policy(state_tensor)
                normal = Normal(action_mean, action_std)
                action = normal.sample()
                log_prob = normal.log_prob(action).sum(dim=-1)
                
                # Apply sigmoid to ensure action is in [0, 1]
                action = torch.sigmoid(action)
            
            # Take action in environment
            next_state, reward, done, _, _ = env.step(action.squeeze().numpy())
            
            # Store experience
            rollouts['states'].append(state)
            rollouts['actions'].append(action.squeeze().numpy())
            rollouts['rewards'].append(reward)
            rollouts['values'].append(value.item())
            rollouts['log_probs'].append(log_prob.item())
            rollouts['dones'].append(done)
            
            # Update state
            state = next_state
            
            if done:
                state, _ = env.reset()
                done = False
        
        # Convert to numpy arrays
        for k, v in rollouts.items():
            rollouts[k] = np.array(v)
        
        # Compute returns and advantages
        rollouts['returns'] = self._compute_returns(rollouts['rewards'], rollouts['dones'])
        rollouts['advantages'] = self._compute_advantages(rollouts['rewards'], rollouts['values'], rollouts['dones'])
        
        return rollouts
    
    def _compute_returns(self, rewards, dones):
        """Compute discounted returns."""
        returns = np.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return
            
        return returns
    
    def _compute_advantages(self, rewards, values, dones):
        """Compute advantages using Generalized Advantage Estimation (GAE)."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        last_value = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * last_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * last_advantage * (1 - dones[t])
            
            last_advantage = advantages[t]
            last_value = values[t]
            
        return advantages
    
    def save(self, path):
        """Save policy network."""
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path):
        """Load policy network."""
        self.policy.load_state_dict(torch.load(path))
