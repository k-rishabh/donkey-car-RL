'''
file: td3_train.py
notes: TD3 implementation with CNN-based Actor and Critic networks for image inputs.
       Includes automatic simulator launch with optional GUI.
       Enhanced with training status logging and TensorBoard integration.

Command lines to run the file:

# For Training:
python td3_train.py --sim /path/DonkeySimLinux/donkey_sim.x86_64 --env_name donkey-generated-roads-v0

# For Testing with GUI:
python td3_train.py --sim /path/DonkeySimLinux/donkey_sim.x86_64 --env_name donkey-generated-roads-v0 --test --gui

# For Testing Headless:
python td3_train.py --sim /path/DonkeySimLinux/donkey_sim.x86_64 --env_name donkey-generated-roads-v0 --test
'''

import argparse
import os
import gym
import gym_donkeycar  # Ensure this import is present
import subprocess
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2  # For image preprocessing
from collections import deque
import random
import traceback
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import copy

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the CNN-based Actor Network
class Actor(nn.Module):
    def __init__(self, state_shape, action_dim, max_action):
        super(Actor, self).__init__()
        # Input shape: (batch_size, channels, height, width)
        self.conv1 = nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of the linear input after convolution layers
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(state_shape[1], 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(state_shape[2], 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(linear_input_size, 512)
        self.out = nn.Linear(512, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = x / 255.0  # Normalize pixel values
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.out(x)
        x = torch.tanh(x) * self.max_action
        return x


# Define the CNN-based Critic Network (Twin Q-networks)
class Critic(nn.Module):
    def __init__(self, state_shape, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.conv1_1 = nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv1_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of the linear input after convolution layers
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(state_shape[1], 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(state_shape[2], 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc1_1 = nn.Linear(linear_input_size + action_dim, 512)
        self.fc1_out = nn.Linear(512, 1)

        # Q2 architecture
        self.conv2_1 = nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv2_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc2_1 = nn.Linear(linear_input_size + action_dim, 512)
        self.fc2_out = nn.Linear(512, 1)

    def forward(self, x, u):
        x = x / 255.0  # Normalize pixel values

        # Q1 forward
        x1 = F.relu(self.conv1_1(x))
        x1 = F.relu(self.conv1_2(x1))
        x1 = F.relu(self.conv1_3(x1))
        x1 = x1.view(x1.size(0), -1)
        x1 = torch.cat([x1, u], 1)
        x1 = F.relu(self.fc1_1(x1))
        x1 = self.fc1_out(x1)

        # Q2 forward
        x2 = F.relu(self.conv2_1(x))
        x2 = F.relu(self.conv2_2(x2))
        x2 = F.relu(self.conv2_3(x2))
        x2 = x2.view(x2.size(0), -1)
        x2 = torch.cat([x2, u], 1)
        x2 = F.relu(self.fc2_1(x2))
        x2 = self.fc2_out(x2)

        return x1, x2

    def Q1(self, x, u):
        x = x / 255.0  # Normalize pixel values

        x1 = F.relu(self.conv1_1(x))
        x1 = F.relu(self.conv1_2(x1))
        x1 = F.relu(self.conv1_3(x1))
        x1 = x1.view(x1.size(0), -1)
        x1 = torch.cat([x1, u], 1)
        x1 = F.relu(self.fc1_1(x1))
        x1 = self.fc1_out(x1)
        return x1


# Define the Replay Buffer
class ReplayBuffer(object):
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


class TD3Agent(object):
    def __init__(
        self,
        state_shape,
        action_dim,
        max_action,
        lr=1e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
    ):
        self.actor = Actor(state_shape, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_shape, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_delay = policy_delay

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample a batch of transitions from replay buffer
        state, next_state, action, reward, done = replay_buffer.sample(batch_size)

        # Select action according to policy and add clipped noise
        noise = (
            torch.randn_like(action) * self.policy_noise
        ).clamp(-self.noise_clip, self.noise_clip)

        next_action = (
            self.actor_target(next_state) + noise
        ).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update the target networks
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        return critic_loss.item(), actor_loss.item() if actor_loss is not None else 0.0


# Function to launch the simulator
def launch_simulator(sim_path, port, gui=False):
    """
    Launches the Donkey Car simulator.

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
                [sim_path, "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            print(f"Launching simulator from {sim_path} on port {port} in headless mode...")
            # Launch simulator with headless flags
            simulator_process = subprocess.Popen(
                [sim_path, "--port", str(port), "-batchmode", "-nographics", "-silent-crashes"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
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


# Function to preprocess the image
def preprocess_state(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    state = cv2.resize(state, (84, 84))  # Resize to (84,84)
    state = np.expand_dims(state, axis=0)  # Add channel dimension
    return state


if __name__ == "__main__":

    # Optional: Setup logging
    logging.basicConfig(
        filename='training.log',  # Log file name
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

    # Set environment variables for gym-donkeycar
    os.environ['DONKEY_SIM_PATH'] = args.sim
    os.environ['DONKEY_SIM_PORT'] = str(args.port)

    # Launch the simulator with or without GUI based on the --gui flag
    simulator_process = launch_simulator(args.sim, args.port, gui=args.gui)

    # Initialize the environment
    env = gym.make(env_id)
    state_shape = (1, 84, 84)  # Grayscale image with channel dimension
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])  # Assuming symmetric action space

    # Initialize the TD3 agent
    agent = TD3Agent(state_shape, action_dim, max_action)

    if args.test:
        # Load the trained actor network
        agent.actor.load_state_dict(torch.load("td3_actor.pth", map_location=device))
        print("Loaded trained actor network.")
        logger.info("Loaded trained actor network for testing.")

        # Run the testing loop
        try:
            while True:
                state = env.reset()
                state = preprocess_state(state)
                done = False
                ep_reward = 0
                ep_timesteps = 0

                while not done:
                    # Select action
                    with torch.no_grad():
                        action = agent.select_action(state)
                    # Clip action to environment's action space
                    action = action.clip(env.action_space.low, env.action_space.high)

                    next_state, reward, done, info = env.step(action)
                    next_state = preprocess_state(next_state)

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

    else:
        # Initialize replay buffer
        replay_buffer = ReplayBuffer()

        # Training parameters
        max_episodes = 10000
        max_timesteps = 2000  # Max timesteps per episode
        start_timesteps = 20000  # Number of timesteps to collect random actions
        batch_size = 256
        exploration_noise = 0.2  # Increased exploration noise
        update_after = start_timesteps  # Number of steps before updating
        update_every = 20    # Update every 'update_every' steps

        episode_rewards = []
        episode_timesteps = []
        total_timesteps = 0

        try:
            for episode in range(1, max_episodes + 1):
                start_time = time.time()  # Start timing the episode
                state = env.reset()
                state = preprocess_state(state)
                done = False
                ep_reward = 0
                ep_timesteps = 0

                for t in range(max_timesteps):
                    total_timesteps += 1
                    ep_timesteps += 1

                    if total_timesteps < start_timesteps:
                        # Select random action
                        action = env.action_space.sample()
                    else:
                        # Select action according to policy with exploration noise
                        action = agent.select_action(state)
                        action = (
                            action
                            + np.random.normal(0, exploration_noise * max_action, size=action_dim)
                        ).clip(env.action_space.low, env.action_space.high)

                    next_state, reward, done, _ = env.step(action)
                    next_state = preprocess_state(next_state)

                    # Store data in replay buffer
                    replay_buffer.add((state, next_state, action, reward, float(done)))

                    state = next_state
                    ep_reward += reward

                    # Train agent after collecting sufficient data
                    if total_timesteps >= update_after and total_timesteps % update_every == 0:
                        for _ in range(update_every):
                            critic_loss, actor_loss = agent.train(replay_buffer, batch_size)
                            # Log losses
                            writer.add_scalar('Loss/Critic', critic_loss, total_timesteps)
                            writer.add_scalar('Loss/Actor', actor_loss, total_timesteps)

                    if done:
                        break

                end_time = time.time()  # End timing the episode
                episode_time = end_time - start_time

                episode_rewards.append(ep_reward)
                episode_timesteps.append(ep_timesteps)

                # Logging
                print(f"Episode {episode}\tTotal Timesteps {total_timesteps}\tEpisode Reward: {ep_reward:.2f}\tTime: {episode_time:.2f} seconds")
                logger.info(f"Episode {episode}\tTotal Timesteps {total_timesteps}\tEpisode Reward: {ep_reward:.2f}")

                # TensorBoard logging
                writer.add_scalar('Episode Reward', ep_reward, episode)
                writer.add_scalar('Episode Timesteps', ep_timesteps, episode)
                writer.add_scalar('Episode Time', episode_time, episode)

        except KeyboardInterrupt:
            print("Training interrupted by user.")
            logger.info("Training interrupted by user.")

        except Exception as e:
            print(f"An error occurred during training: {e}")
            logger.error(f"An error occurred during training: {e}")
            traceback.print_exc()

        finally:
            # Save the final model
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
