import cv2
import argparse
from itertools import count
import os
import random
import numpy as np
from collections import deque

import gym
import gym_donkeycar
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
parser.add_argument("--env_name", default="donkey-generated-roads-v0", type=str)
parser.add_argument('--tau', default=0.005, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--capacity', default=100000, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--noise_scale', default=0.1, type=float)
parser.add_argument('--max_episode', default=1000, type=int)
parser.add_argument('--max_timestep', default=1000, type=int)
parser.add_argument('--update_iteration', default=200, type=int)
parser.add_argument('--log_interval', default=1, type=int)
parser.add_argument('--render', action='store_true')
parser.add_argument('--save_dir', default='./ddpg_models', type=str)
parser.add_argument('--gui', action='store_true')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize environment
env = gym.make(args.env_name)

state_dim = 84 * 84  # Flattened grayscale image
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

class OrnsteinUhlenbeckNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

class ReplayBuffer:
    def __init__(self, max_size=args.capacity):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, next_state, action, reward, done):
        self.buffer.append((state, next_state, action, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, next_states, actions, rewards, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(next_states),
            np.array(actions),
            np.array(rewards).reshape(-1, 1),
            np.array(dones).reshape(-1, 1)
        )

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.bn_input = nn.BatchNorm1d(state_dim)
        
        self.l1 = nn.Linear(state_dim, 400)
        self.bn1 = nn.BatchNorm1d(400)
        
        self.l2 = nn.Linear(400, 300)
        self.bn2 = nn.BatchNorm1d(300)
        
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.bn_input(x)
        
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        x = self.max_action * torch.tanh(self.l3(x))
        
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.bn_input = nn.BatchNorm1d(state_dim)
        
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.bn1 = nn.BatchNorm1d(400)
        
        self.l2 = nn.Linear(400, 300)
        self.bn2 = nn.BatchNorm1d(300)
        
        self.l3 = nn.Linear(300, 1)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        state = self.bn_input(state)
        x = torch.cat([state, action], dim=1)
        
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        x = self.l3(x)
        
        return x

class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.learning_rate)

        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.9)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.9)

        self.replay_buffer = ReplayBuffer()
        self.writer = SummaryWriter(args.save_dir)

        self.noise = OrnsteinUhlenbeckNoise(action_dim)
        self.num_critic_updates = 0
        self.num_actor_updates = 0

    def select_action(self, state):
        self.actor.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = self.actor(state).cpu().numpy().squeeze()
        self.actor.train()
        return action

    def train(self):
        if len(self.replay_buffer) < args.batch_size:
            return

        for _ in range(args.update_iteration):
            # Sample from replay buffer
            states, next_states, actions, rewards, dones = self.replay_buffer.sample(args.batch_size)
            
            states = torch.FloatTensor(states).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            actions = torch.FloatTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            dones = torch.FloatTensor(1 - dones).to(device)

            # Compute target Q value
            with torch.no_grad():
                next_actions = self.actor_target(next_states)
                target_q = rewards + dones * args.gamma * self.critic_target(next_states, next_actions)

            # Update critic
            current_q = self.critic(states, actions)
            critic_loss = F.mse_loss(current_q, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()

            # Update actor
            actor_loss = -self.critic(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # Log metrics
            self.num_critic_updates += 1
            self.num_actor_updates += 1
            self.writer.add_scalar('Loss/critic', critic_loss.item(), self.num_critic_updates)
            self.writer.add_scalar('Loss/actor', actor_loss.item(), self.num_actor_updates)
            self.writer.add_scalar('Value/average_q', current_q.mean().item(), self.num_critic_updates)

        # Update learning rates
        self.actor_scheduler.step()
        self.critic_scheduler.step()

def preprocess_state(state):
    """Convert RGB image to grayscale and resize to 84x84."""
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, (84, 84))
    return state.flatten().astype(np.float32) / 255.0

def main():
    agent = DDPG(state_dim, action_dim, max_action)
    
    # Load models if testing
    if args.test:
        agent.actor.load_state_dict(torch.load(f"{args.save_dir}/actor.pth", map_location=device))
        agent.critic.load_state_dict(torch.load(f"{args.save_dir}/critic.pth", map_location=device))
    
    for episode in range(args.max_episode):
        state = env.reset()
        state = preprocess_state(state)
        episode_reward = 0
        agent.noise.reset()

        for t in range(args.max_timestep):
            if args.render or args.gui:
                env.render()

            # Select action
            action = agent.select_action(state)
            if not args.test:  # Add noise only during training
                noise = agent.noise.sample()
                action = (action + noise).clip(env.action_space.low, env.action_space.high)

            # Execute action
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)

            if not args.test:
                # Store transition in replay buffer
                agent.replay_buffer.push(state, next_state, action, reward, done)
                # Train agent
                if len(agent.replay_buffer) > args.batch_size:
                    agent.train()

            state = next_state
            episode_reward += reward

            if done:
                break

        print(f"Episode: {episode}, Reward: {episode_reward}")

        # Save models periodically
        if not args.test and episode % args.log_interval == 0:
            torch.save(agent.actor.state_dict(), f"{args.save_dir}/actor.pth")
            torch.save(agent.critic.state_dict(), f"{args.save_dir}/critic.pth")
            agent.writer.add_scalar('Reward/episode', episode_reward, episode)

if __name__ == "__main__":
    main()