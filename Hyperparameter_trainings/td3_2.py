# td3_train.py

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
import random

import cv2  # For image preprocessing

from collections import deque
import traceback

# Optional: For file logging
import logging

# Optional: For TensorBoard integration
from torch.utils.tensorboard import SummaryWriter

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.output(x)) * self.max_action
        return x

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1 architecture
        self.fc1_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_output = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.fc2_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_output = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        
        # Q1 forward
        x1 = F.relu(self.fc1_1(xu))
        x1 = F.relu(self.fc1_2(x1))
        q1 = self.q1_output(x1)
        
        # Q2 forward
        x2 = F.relu(self.fc2_1(xu))
        x2 = F.relu(self.fc2_2(x2))
        q2 = self.q2_output(x2)
        
        return q1, q2
    
    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.fc1_1(xu))
        x1 = F.relu(self.fc1_2(x1))
        q1 = self.q1_output(x1)
        return q1

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.storage = deque(maxlen=max_size)
        
    def add(self, state, action, next_state, reward, done):
        self.storage.append((state, action, next_state, reward, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        state, action, next_state, reward, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(reward).unsqueeze(1).to(device),
            torch.FloatTensor(done).unsqueeze(1).to(device)
        )

# TD3 Agent
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, gamma=0.99, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_delay=2, hidden_dim=256):
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_delay = policy_delay
        self.max_action = max_action
        
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.total_it = 0  # Keep track of the total number of updates
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size=64):
        self.total_it += 1
        
        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
            # Compute the target Q-value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update the target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Function to launch the simulator (same as PPO code)
def launch_simulator(sim_path, port, gui=False):
    """
    Launches the Donkey Car simulator on Windows.

    :param sim_path: (str) Absolute path to the simulator executable.
    :param port: (int) Port number for TCP communication.
    :param gui: (bool) Whether to launch the simulator with GUI.
    :return: subprocess.Popen object representing the simulator process.
    """
    try:
        if gui:
            print(f"Launching simulator from {sim_path} on port {port} with GUI...")
            # Launch simulator without headless flags
            simulator_process = subprocess.Popen(
                f'"{sim_path}" --port {port}',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )
        else:
            print(f"Launching simulator from {sim_path} on port {port} in headless mode...")
            # Launch simulator with headless flag if supported
            simulator_process = subprocess.Popen(
                f'"{sim_path}" --port {port} --headless',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )
        
        # Wait for the simulator to initialize
        print("Waiting for simulator to initialize...")
        time.sleep(10)  # Adjust if the simulator takes longer to start
        print(f"Simulator launched successfully {'with GUI' if gui else 'in headless mode'}.")
        return simulator_process
    except Exception as e:
        print(f"Failed to launch simulator: {e}")
        sys.exit(1)

# Function to terminate the simulator
def terminate_simulator(simulator_process):
    """
    Terminates the Donkey Car simulator subprocess.

    :param simulator_process: subprocess.Popen object representing the simulator process.
    """
    try:
        print("Terminating simulator...")
        simulator_process.terminate()
        simulator_process.wait(timeout=5)
        print("Simulator terminated.")
    except Exception as e:
        print(f"Error terminating simulator: {e}")

# Function to scale actions
def scale_action(action):
    action = np.clip(action, -1, 1)
    return action  # Since env expects actions in [-1, 1]

# Main function
if __name__ == "__main__":
    # Optional: Setup logging
    logging.basicConfig(
        filename='training_td3.log',  # Log file name
        filemode='a',              # Append mode
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger()

    # Optional: Initialize TensorBoard writer
    writer = SummaryWriter('runs/td3_training')

    # Environment configuration
    env_list = [
        "donkey-warehouse-v0",
        "donkey-generated-roads-v0",
        "donkey-avc-sparkfun-v0",
        "donkey-generated-track-v0",
        "donkey-roboracingleague-track-v0",
        "donkey-waveshare-v0",
        "donkey-minimonaco-track-v0",
        "donkey-warren-track-v0",
        "donkey-thunderhill-track-v0",
        "donkey-circuit-launch-track-v0",
    ]

    parser = argparse.ArgumentParser(description="td3_train")
    parser.add_argument(
        "--sim",
        type=str,
        required=True,
        help="Path to the simulator executable.",
    )
    parser.add_argument("--port", type=int, default=9091, help="Port to use for TCP.")
    parser.add_argument("--test", action="store_true", help="Load the trained model and play.")
    parser.add_argument(
        "--env_name",
        type=str,
        default="donkey-generated-roads-v0",
        help="Name of the Donkey Sim environment.",
        choices=env_list,
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch simulator with GUI. If not set, runs in headless mode."
    )

    args = parser.parse_args()

    env_id = args.env_name

    # Adjust the simulator path for Windows
    sim_path = args.sim

    # Set environment variables for gym-donkeycar
    os.environ['DONKEY_SIM_PATH'] = sim_path
    os.environ['DONKEY_SIM_PORT'] = str(args.port)

    # Launch the simulator with or without GUI based on the --gui flag
    simulator_process = launch_simulator(sim_path, args.port, gui=args.gui)

    # Initialize the environment
    env = gym.make(env_id)
    state_dim = 84 * 84  # After preprocessing, the state is a flattened (84,84) image
    action_dim = env.action_space.shape[0]
    max_action = 1.0  # Since actions are in [-1, 1]

    # Initialize the TD3 agent
    agent = TD3Agent(state_dim, action_dim, max_action)

    if args.test:
        # Load the trained actor network
        agent.actor.load_state_dict(torch.load("td3_actor.pth", map_location=device))
        print("Loaded trained actor network.")
        logger.info("Loaded trained actor network for testing.")

        # Run the testing loop
        try:
            while True:
                state = env.reset()
                done = False
                ep_reward = 0
                ep_timesteps = 0

                while not done:
                    # Preprocess state
                    state_preprocessed = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
                    state_preprocessed = cv2.resize(state_preprocessed, (84, 84))  # Resize to reduce dimensionality
                    state_preprocessed = state_preprocessed.flatten() / 255.0      # Flatten and normalize

                    # Select action without exploration
                    action = agent.select_action(state_preprocessed)
                    # Scale action to environment's action space
                    scaled_action = scale_action(action)
                    scaled_action = scaled_action.clip(env.action_space.low, env.action_space.high)

                    next_state, reward, done, info = env.step(scaled_action)

                    ep_reward += reward
                    ep_timesteps += 1

                    state = next_state

                    if done:
                        print(f"Test Episode Reward: {ep_reward}\tTimesteps: {ep_timesteps}")
                        logger.info(f"Test Episode Reward: {ep_reward}\tTimesteps: {ep_timesteps}")
                        break
        except KeyboardInterrupt:
            print("Testing interrupted by user.")
            logger.info("Testing interrupted by user.")
        except Exception as e:
            print(f"An error occurred during testing: {e}")
            logger.error(f"An error occurred during testing: {e}")
            traceback.print_exc()
        finally:
            # Close the environment
            env.close()
            # Terminate the simulator
            terminate_simulator(simulator_process)
            # Close TensorBoard writer
            writer.close()
            sys.exit()

    # Initialize replay buffer
    replay_buffer = ReplayBuffer()

    # Training parameters
    max_episodes = 1000
    max_timesteps = 2000
    exploration_noise = 0.1
    batch_size = 64
    start_timesteps = 10000  # Number of timesteps before starting training

    timestep = 0
    episode = 0

    # Initialize metrics
    episode_rewards = []
    episode_timesteps = []

    try:
        while episode < max_episodes:
            state = env.reset()
            done = False
            ep_reward = 0
            ep_timesteps = 0

            for t in range(max_timesteps):
                # Preprocess state
                state_preprocessed = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
                state_preprocessed = cv2.resize(state_preprocessed, (84, 84))  # Resize to reduce dimensionality
                state_preprocessed = state_preprocessed.flatten() / 255.0      # Flatten and normalize

                if timestep < start_timesteps:
                    # Select random action
                    action = env.action_space.sample()
                else:
                    # Select action according to policy and add exploration noise
                    action = agent.select_action(state_preprocessed)
                    action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(-max_action, max_action)

                # Scale action to environment's action space
                scaled_action = scale_action(action)
                scaled_action = scaled_action.clip(env.action_space.low, env.action_space.high)

                next_state, reward, done, _ = env.step(scaled_action)

                # Preprocess next state
                next_state_preprocessed = cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY)
                next_state_preprocessed = cv2.resize(next_state_preprocessed, (84, 84))
                next_state_preprocessed = next_state_preprocessed.flatten() / 255.0

                # Store data in replay buffer
                replay_buffer.add(state_preprocessed, action, next_state_preprocessed, reward, done)

                state = next_state
                ep_reward += reward
                ep_timesteps += 1
                timestep += 1

                # Train agent after collecting sufficient data
                if timestep >= start_timesteps:
                    agent.train(replay_buffer, batch_size)

                if done:
                    break

            episode += 1
            episode_rewards.append(ep_reward)
            episode_timesteps.append(ep_timesteps)

            # Logging
            if episode % 1 == 0:
                avg_reward = np.mean(episode_rewards[-1:])
                avg_timesteps = np.mean(episode_timesteps[-1:])

                print(f"Episode {episode}\tTimestep {timestep}")
                print(f"Episode Reward: {ep_reward:.2f}\tTimesteps: {ep_timesteps}")
                logger.info(f"Episode {episode}\tTimestep {timestep}")
                logger.info(f"Episode Reward: {ep_reward:.2f}\tTimesteps: {ep_timesteps}")

                # Optional: Log to TensorBoard
                writer.add_scalar('Episode Reward', ep_reward, episode)
                writer.add_scalar('Episode Timesteps', ep_timesteps, episode)

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        logger.info("Training interrupted by user.")

    except Exception as e:
        print(f"An error occurred during training at episode {episode}, timestep {timestep}: {e}")
        logger.error(f"An error occurred during training at episode {episode}, timestep {timestep}: {e}")
        traceback.print_exc()

    finally:
        # Save the final models
        torch.save(agent.actor.state_dict(), "td3_actor.pth")
        torch.save(agent.critic.state_dict(), "td3_critic.pth")
        print("Models saved.")
        logger.info("Models saved.")

        # Close the environment
        env.close()
        logger.info("Environment closed.")

        # Terminate the simulator
        terminate_simulator(simulator_process)
        logger.info("Simulator terminated.")

        # Close TensorBoard writer
        writer.close()
        logger.info("TensorBoard writer closed.")
