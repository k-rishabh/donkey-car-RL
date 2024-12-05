import cv2
import argparse
from itertools import count
import os, random
import numpy as np

import gym
import gym_donkeycar
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)  # 'train' or 'test'
parser.add_argument("--env_name", default="donkey-generated-roads-v0", type=str)
parser.add_argument('--tau', default=0.005, type=float)  # Target smoothing coefficient
parser.add_argument('--gamma', default=0.99, type=float)  # Discount factor
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--capacity', default=100000, type=int)  # Replay buffer size
parser.add_argument('--batch_size', default=64, type=int)  # Mini-batch size
parser.add_argument('--exploration_noise', default=0.1, type=float)  # Noise for exploration
parser.add_argument('--theta', default=0.15, type=float)  # OU noise theta
parser.add_argument('--sigma', default=0.2, type=float)  # OU noise sigma
parser.add_argument('--max_episode', default=1000, type=int)  # Number of episodes
parser.add_argument('--max_timestep', default=1000, type=int)  # Max timesteps per episode
parser.add_argument('--update_iteration', default=200, type=int)  # Updates per training loop
parser.add_argument('--log_interval', default=1, type=int)
parser.add_argument('--render', action='store_true')  # Render environment
parser.add_argument('--save_dir', default='./ddpg_models', type=str)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize environment
env = gym.make(args.env_name)
state_dim = 84 * 84  # Input is preprocessed camera images (grayscale, flattened)
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[i] for i in indices]
        states, next_states, actions, rewards, dones = zip(*batch)
        return (
            np.array(states),
            np.array(next_states),
            np.array(actions),
            np.array(rewards).reshape(-1, 1),
            np.array(dones).reshape(-1, 1),
        )

# OU Noise
class OUNoise:
    def __init__(self, action_dim, theta=0.15, sigma=0.2, dt=1e-2, mu=0.0):
        self.action_dim = action_dim
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.mu = mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.state += dx
        return self.state

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(7 * 7 * 64, 256)  # Flattened output size depends on input image size
        self.fc2 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = x.view(-1, 1, 84, 84)  # Reshape to [batch_size, channels, height, width]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.max_action * torch.tanh(self.fc2(x))
        return x


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
         # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(7 * 7 * 64 + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        state = state.view(-1, 1, 84, 84)  # Reshape to [batch_size, channels, height, width]
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.cat([x, action], dim=1)  # Concatenate state features with action
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# DDPG Agent
class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.learning_rate)

        self.replay_buffer = ReplayBuffer()
        self.noise = OUNoise(action_dim, theta=args.theta, sigma=args.sigma)
        self.writer = SummaryWriter(args.save_dir)

        self.num_critic_updates = 0
        self.num_actor_updates = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().detach().numpy()

    def train(self):
        for _ in range(args.update_iteration):
            states, next_states, actions, rewards, dones = self.replay_buffer.sample(args.batch_size)

            states = torch.FloatTensor(states).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            actions = torch.FloatTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            dones = torch.FloatTensor(1 - dones).to(device)

            # Critic update
            target_q = rewards + (dones * args.gamma * self.critic_target(next_states, self.actor_target(next_states)))
            current_q = self.critic(states, actions)
            critic_loss = F.mse_loss(current_q, target_q.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor update
            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Target networks update
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)


# Preprocessing
def preprocess_state(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, (84, 84))
    return state.astype(np.float32) / 255.0  # Keep the image as 2D for convolutional layers



# Main Training Loop
if __name__ == "__main__":
    agent = DDPG(state_dim, action_dim, max_action)

    if args.mode == "train":
        for episode in range(args.max_episode):
        
            state = preprocess_state(env.reset())
            agent.noise.reset() # Reset OU noise at the start of each episode
            episode_reward = 0
            for t in range(args.max_timestep):
                action = agent.select_action(state)
                action += agent.noise.sample()  # Add OU noise for exploration
                action = np.clip(action, env.action_space.low, env.action_space.high)
                action = np.squeeze(action)  # Ensure action has the correct shape

                #print("Action Shape:", action.shape)
                #print("Action Space:", env.action_space)
                #print("Expected Action Dim:", action_dim)
                next_state, reward, done, _ = env.step(action)
                next_state = preprocess_state(next_state)

                agent.replay_buffer.push((state, next_state, action, reward, done))
                state = next_state
                episode_reward += reward

                if len(agent.replay_buffer.storage) > args.batch_size:
                    agent.train()

                if done:
                    break

            print(f"Episode {episode}, Reward: {episode_reward}")

            if episode % args.log_interval == 0:
                torch.save(agent.actor.state_dict(), f"{args.save_dir}/actor.pth")
                torch.save(agent.critic.state_dict(), f"{args.save_dir}/critic.pth")