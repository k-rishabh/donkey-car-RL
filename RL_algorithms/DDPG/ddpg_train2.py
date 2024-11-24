'''
file: ddpg_train.py
notes: DDPG implementation for continuous action spaces.
       Includes automatic simulator launch with optional GUI.
python ddpg_train.py --sim /path/donkey_sim.exe --env_name donkey-generated-roads-v0   #(FOR TRAINING)
python ddpg_train.py --sim /path/donkey_sim.exe --env_name donkey-generated-roads-v0 --test --gui      # (FOR TESTING GUI)
python ddpg_train.py --sim /path/donkey_sim.exe --env_name donkey-generated-roads-v0 --test     # (FOR TESTING HEADLESS) 
'''

import argparse
import os
import gym
import gym_donkeycar
import subprocess
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))

# Define the Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lr=1e-3, gamma=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=64):
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        
        # Convert to tensors
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device)
        
        # Critic loss
        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + (1 - done) * self.gamma * target_q
        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)

        # Actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, next_states, rewards, dones = zip(*[self.buffer[i] for i in idx])
        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards), np.array(dones)

    def size(self):
        return len(self.buffer)

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ddpg_train")
    parser.add_argument("--sim", type=str, required=True, help="Path to the simulator executable.")
    parser.add_argument("--port", type=int, default=9091, help="Port to use for TCP.")
    parser.add_argument("--test", action="store_true", help="Load the trained model and play.")
    parser.add_argument("--env_name", type=str, default="donkey-generated-roads-v0", help="Environment name.")
    parser.add_argument("--gui", action="store_true", help="Launch simulator with GUI.")
    args = parser.parse_args()

    # Launch simulator
    simulator_process = launch_simulator(args.sim, args.port, gui=args.gui)

    # Initialize environment
    env = gym.make(args.env_name)
    state_dim = 84 * 84  # Preprocessed state size
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()
    writer = SummaryWriter('runs/ddpg_training')

    if args.test:
        agent.actor.load_state_dict(torch.load("ddpg_donkey_actor.pth"))
        # Testing loop (similar to PPO)
    else:
        # Training loop (similar to PPO)
