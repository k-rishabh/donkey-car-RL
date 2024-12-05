import argparse
import os
import gym
import gym_donkeycar 
import subprocess
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2 
from collections import deque
import traceback
import logging
from helper import *

# Optional: For TensorBoard integration
from torch.utils.tensorboard import SummaryWriter

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.mean.weight.data.mul_(0.1)
        self.mean.bias.data.mul_(0.0)
        self.log_std.weight.data.mul_(0.1)
        self.log_std.bias.data.mul_(0.0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Stability
        
        std = torch.exp(log_std)
        return mean, std

# Define the Value Network
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        state_value = self.value_head(x)
        return state_value

# Define the Memory Class
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []           # Pre-tanh actions
        self.actions_tanh = []      # Tanh-transformed actions
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = []
        self.returns = []
    
    def clear(self):
        self.states = []
        self.actions = []
        self.actions_tanh = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = []
        self.returns = []

# Define the PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95, clip_epsilon=0.2, 
                 K_epochs=4, minibatch_size=64, entropy_coef=0.01):
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.K_epochs = K_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef

        self.policy = Actor(state_dim, action_dim).to(device)
        self.policy_old = Actor(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.value_net = Critic(state_dim).to(device)
        
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_net.parameters(), 'lr': lr}
        ])
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        mean, std = self.policy_old(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()  # Allows gradient flow
        action_tanh = torch.tanh(action)
        # Compute log probability with correction
        logprob = dist.log_prob(action) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        logprob = logprob.sum(dim=-1)
        action_tanh = action_tanh.detach().cpu().numpy()
        logprob = logprob.detach().cpu().numpy()
        action = action.detach().cpu().numpy()  # Pre-tanh action
        return action_tanh, logprob, action  # Return tanh action, logprob, and pre-tanh action
    
    def compute_gae(self, rewards, dones, values):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        advantages = np.array(returns) - values[:-1]
        return returns, advantages
    
    def update(self, memory):
        # Convert lists to tensors
        old_states = torch.FloatTensor(np.array(memory.states)).to(device)
        old_actions = torch.FloatTensor(np.array(memory.actions)).to(device)        # Pre-tanh actions
        old_logprobs = torch.FloatTensor(np.array(memory.logprobs)).to(device)
        old_returns = torch.FloatTensor(np.array(memory.returns)).to(device)
        old_advantages = torch.FloatTensor(np.array(memory.advantages)).to(device)
        
        # Normalize advantages
        old_advantages = (old_advantages - old_advantages.mean()) / (old_advantages.std() + 1e-8)
        
        dataset = torch.utils.data.TensorDataset(old_states, old_actions, old_logprobs, old_returns, old_advantages)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)
        
        for _ in range(self.K_epochs):
            for batch in dataloader:
                states, actions, logprobs, returns, advantages = batch
                
                # Evaluating old actions and values
                mean, std = self.policy(states)
                dist = torch.distributions.Normal(mean, std)
                new_logprobs = dist.log_prob(actions) - torch.log(1 - torch.tanh(actions).pow(2) + 1e-6)
                new_logprobs = new_logprobs.sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Finding the ratio (pi_theta / pi_theta_old)
                ratios = torch.exp(new_logprobs - logprobs)
                
                # Finding Surrogate Loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Finding Value Loss
                state_values = self.value_net(states).squeeze()
                value_loss = self.MseLoss(state_values, returns)
                
                # Total Loss
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                
                # Take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

