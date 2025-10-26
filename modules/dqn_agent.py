"""
modules/dqn_agent.py - DQN Agent (FIXED)

REPLACE your existing modules/dqn_agent.py with this fixed version.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class QNetwork(nn.Module):
    """Neural network for Q-values."""
    
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for humanoid control."""
    
    def __init__(self, state_dim, num_joints=4, actions_per_joint=5,
                 learning_rate=1e-4, epsilon_start=1.0, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQN] Using device: {self.device}")
        
        # Action space
        self.num_joints = num_joints
        self.actions_per_joint = actions_per_joint
        self.total_actions = num_joints * actions_per_joint
        self.torque_bins = np.linspace(-1.0, 1.0, actions_per_joint)
        
        # Networks
        self.q_network = QNetwork(state_dim, self.total_actions).to(self.device)
        self.target_network = QNetwork(state_dim, self.total_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = epsilon_start
        self.epsilon_end = 0.05
        self.epsilon_decay = epsilon_decay
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 64
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy."""
        if training and random.random() < self.epsilon:
            action_indices = np.random.randint(0, self.actions_per_joint, 
                                              size=self.num_joints)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            q_values = q_values.view(self.num_joints, self.actions_per_joint)
            action_indices = q_values.argmax(dim=1).cpu().numpy()
        
        torques = self.torque_bins[action_indices] * -10 # Scale to [-0.8, 0.8]
        return action_indices, torques
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states)
        current_q_values = current_q_values.view(-1, self.num_joints, 
                                                  self.actions_per_joint)
        
        # Gather Q-values for taken actions (FIXED)
        batch_size = current_q_values.shape[0]
        current_q = torch.zeros(batch_size, self.num_joints).to(self.device)
        
        for i in range(self.num_joints):
            # Get the action index for joint i for all samples in batch
            joint_actions = actions[:, i]  # Shape: [batch_size]
            # Gather Q-values for those actions
            current_q[:, i] = current_q_values[:, i, :].gather(1, joint_actions.unsqueeze(1)).squeeze(1)
        
        # Average Q-value across joints
        current_q = current_q.mean(dim=1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.view(-1, self.num_joints,
                                               self.actions_per_joint)
            next_q = next_q_values.max(dim=2)[0].mean(dim=1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save model."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"[DQN] Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"[DQN] Model loaded from {filepath}")