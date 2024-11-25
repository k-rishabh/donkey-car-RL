import argparse
import os
import gym
import gym_donkeycar
import torch
import numpy as np
import cv2
from collections import deque
import logging
import subprocess
import time
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
    filename='testing.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

# Function to preprocess the frames
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    frame = frame.astype(np.float32) / 255.0
    return np.expand_dims(frame, axis=0)

# Function to launch the simulator
def launch_simulator(sim_path, port, gui):
    args = [sim_path, "--port", str(port)]
    if not gui:
        args += ["-batchmode", "-nographics", "-silent-crashes"]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Function to terminate the simulator
def terminate_simulator(simulator_process):
    simulator_process.terminate()
    simulator_process.wait()

# TD3 Actor Model
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

def test_td3(args):
    # Launch simulator
    simulator_process = launch_simulator(args.sim, args.port, args.gui)

    try:
        # Initialize the environment
        env = gym.make(args.env_name)

        # Load trained model
        state_shape = (1, 84, 84)  # Grayscale frame dimensions
        action_dim = env.action_space.shape[0]
        action_space_low = env.action_space.low
        action_space_high = env.action_space.high

        actor = Actor(state_shape, action_dim, action_space_low, action_space_high).to(device)
        actor.load_state_dict(torch.load(args.model_path, map_location=device))
        actor.eval()
        logger.info("Loaded trained model for testing.")

        total_rewards = []
        for episode in range(1, args.episodes + 1):
            state = preprocess_frame(env.reset())
            total_reward = 0
            done = False
            timesteps = 0

            while not done:
                with torch.no_grad():
                    action = actor(torch.FloatTensor(state).unsqueeze(0).to(device)).cpu().numpy().flatten()

                next_state, reward, done, info = env.step(action)
                state = preprocess_frame(next_state)
                total_reward += reward
                timesteps += 1

                if done:
                    break

            total_rewards.append(total_reward)
            print(f"Episode {episode}: Total Reward = {total_reward:.2f}")
            logger.info(f"Episode {episode}: Total Reward = {total_reward:.2f}, Timesteps = {timesteps}")

        print(f"Average Reward over {args.episodes} episodes: {np.mean(total_rewards):.2f}")
        logger.info(f"Average Reward over {args.episodes} episodes: {np.mean(total_rewards):.2f}")

    except Exception as e:
        print(f"An error occurred during testing: {e}")
        logger.error(f"An error occurred during testing: {e}")
    finally:
        # Terminate simulator
        terminate_simulator(simulator_process)
        logger.info("Simulator terminated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TD3 Testing Script")
    parser.add_argument("--sim", type=str, required=True, help="Path to the simulator executable.")
    parser.add_argument("--port", type=int, default=9091, help="Simulator port.")
    parser.add_argument("--env_name", type=str, default="donkey-generated-roads-v0", help="Environment name.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained actor model.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes for testing.")
    parser.add_argument("--gui", action="store_true", help="Enable GUI.")

    args = parser.parse_args()
    test_td3(args)
