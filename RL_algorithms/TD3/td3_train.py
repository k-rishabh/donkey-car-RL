'''
python td3_train.py --sim "path\to\donkey_sim\donkey_sim.exe" --port 9091 (TRAIN)
'''
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
import numpy as np
import cv2
from collections import deque
import random
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import copy

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging setup
logging.basicConfig(
    filename='training.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

# TensorBoard Writer
writer = SummaryWriter('runs/td3_training')

class Actor(nn.Module):
    def __init__(self, state_shape, action_dim, action_space_low, action_space_high):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 256)

        self.steering_head = nn.Linear(256, 1)
        self.throttle_head = nn.Linear(256, 1)

        self.action_space_low = torch.FloatTensor(action_space_low).to(device)
        self.action_space_high = torch.FloatTensor(action_space_high).to(device)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        steering = torch.tanh(self.steering_head(x))
        throttle = torch.sigmoid(self.throttle_head(x))

        action = torch.cat([steering, throttle], dim=1)
        return action * (self.action_space_high - self.action_space_low) / 2 + \
               (self.action_space_high + self.action_space_low) / 2


class Critic(nn.Module):
    def __init__(self, state_shape, action_dim):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(3136 + action_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, 1)

    def forward(self, x, action):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.storage = deque(maxlen=max_size)

    def add(self, data):
        self.storage.append(data)

    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        state, next_state, action, reward, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(state)).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(np.array(action)).to(device),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(done)).unsqueeze(1).to(device),
        )

class TD3Agent:
    def __init__(self, state_shape, action_dim, action_space_low, action_space_high, lr=1e-4, gamma=0.99, tau=0.005):
        self.actor = Actor(state_shape, action_dim, action_space_low, action_space_high).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_shape, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.action_space_low = torch.FloatTensor(action_space_low).to(device)
        self.action_space_high = torch.FloatTensor(action_space_high).to(device)

        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        state, next_state, action, reward, done = replay_buffer.sample(batch_size)

        # Compute targets
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = reward + self.gamma * (1 - done) * self.critic_target(next_state, next_action)

        # Update critic
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    frame = frame.astype(np.float32) / 255.0
    return np.expand_dims(frame, axis=0)


def launch_simulator(sim_path, port, gui):
    args = [sim_path, "--port", str(port)]
    if not gui:
        args += ["-batchmode", "-nographics", "-silent-crashes"]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def terminate_simulator(simulator_process):
    simulator_process.terminate()
    simulator_process.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, required=True, help="Path to simulator executable.")
    parser.add_argument("--port", type=int, default=9091, help="Simulator port.")
    parser.add_argument("--gui", action="store_true", help="Enable GUI.")
    parser.add_argument("--test", action="store_true", help="Test trained model.")
    args = parser.parse_args()

    simulator_process = launch_simulator(args.sim, args.port, gui=args.gui)
    env = gym.make("donkey-generated-roads-v0")

    state_shape = (1, 84, 84)
    action_dim = env.action_space.shape[0]
    agent = TD3Agent(state_shape, action_dim, env.action_space.low, env.action_space.high)

    replay_buffer = ReplayBuffer()
    for episode in range(1, 1001):
        state = preprocess_frame(env.reset())
        total_reward = 0

        for _ in range(1000):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, preprocess_frame(next_state), action, reward, done))
            state = preprocess_frame(next_state)
            total_reward += reward

            if len(replay_buffer.storage) > 256:
                agent.train(replay_buffer)

            if done:
                break

        print(f"Episode {episode}, Total Reward: {total_reward}")

        if episode % 100 == 0:
            torch.save(agent.actor.state_dict(), f"td3_actor_{episode}.pth")
            torch.save(agent.critic.state_dict(), f"td3_critic_{episode}.pth")
            print(f"Models saved at episode {episode}.")
