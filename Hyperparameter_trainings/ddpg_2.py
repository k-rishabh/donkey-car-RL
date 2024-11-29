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
import logging
import random
from collections import deque
import traceback
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Import matplotlib for plotting
import matplotlib.pyplot as plt

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def size(self):
        return len(self.buffer)

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.max_action * torch.tanh(self.out(x))
        return x

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        x1 = self.out(x1)
        return x1

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self, replay_buffer, batch_size=100):
        # Sample replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        # Compute the target Q value
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = reward + ((1 - done) * self.gamma * self.critic_target(next_state, next_action))

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Function to launch the simulator
def launch_simulator(sim_path, port, gui=False):
    try:
        if gui:
            print(f"Launching simulator from {sim_path} on port {port} with GUI...")
            simulator_process = subprocess.Popen([
                sim_path,
                "--port", str(port)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            print(f"Launching simulator from {sim_path} on port {port} in headless mode...")
            simulator_process = subprocess.Popen([
                sim_path,
                "--port", str(port),
                "-batchmode",
                "-nographics",
                "-silent-crashes"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Waiting for simulator to initialize...")
        time.sleep(10)
        print(f"Simulator launched successfully {'with GUI' if gui else 'in headless mode'}.")
        return simulator_process
    except Exception as e:
        print(f"Failed to launch simulator: {e}")
        sys.exit(1)

# Function to terminate the simulator
def terminate_simulator(simulator_process):
    try:
        print("Terminating simulator...")
        simulator_process.terminate()
        simulator_process.wait(timeout=5)
        print("Simulator terminated.")
    except Exception as e:
        print(f"Error terminating simulator: {e}")

# Main function
if __name__ == "__main__":

    # Setup logging
    logging.basicConfig(
        filename='training.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger()

    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/ddpg_training')

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

    parser = argparse.ArgumentParser(description="ddpg_train")
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

    # Set environment variables and launch simulator
    os.environ['DONKEY_SIM_PATH'] = args.sim
    os.environ['DONKEY_SIM_PORT'] = str(args.port)

    simulator_process = launch_simulator(args.sim, args.port, gui=args.gui)

    # Initialize the environment
    env = gym.make(env_id)
    state_dim = 84 * 84  # After preprocessing
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize the DDPG agent
    agent = DDPGAgent(state_dim, action_dim, max_action)

    # Initialize the replay buffer
    replay_buffer = ReplayBuffer()

    # Training parameters
    max_episodes = 1000
    max_timesteps = 1000
    batch_size = 100
    exploration_noise = 0.15
    start_timesteps = 20000
    log_interval = 1

    timestep = 0
    episode = 0

    # Initialize metrics
    episode_rewards = []
    episode_timesteps = []

    # Initialize best average reward
    best_avg_reward = float('-inf')

    try:
        while episode < max_episodes:
            state = env.reset()
            done = False
            ep_reward = 0
            ep_timesteps = 0

            while not done:
                # Preprocess state
                state_preprocessed = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
                state_preprocessed = cv2.resize(state_preprocessed, (84, 84))
                state_preprocessed = state_preprocessed.flatten() / 255.0

                if timestep < start_timesteps:
                    # Select random action
                    action = env.action_space.sample()
                else:
                    # Select action according to policy
                    action = agent.select_action(state_preprocessed)
                    # Add exploration noise
                    action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(
                        env.action_space.low, env.action_space.high
                    )

                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                ep_timesteps += 1

                # Preprocess next state
                next_state_preprocessed = cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY)
                next_state_preprocessed = cv2.resize(next_state_preprocessed, (84, 84))
                next_state_preprocessed = next_state_preprocessed.flatten() / 255.0

                # Store data in replay buffer
                replay_buffer.add(state_preprocessed, action, reward, next_state_preprocessed, float(done))

                state = next_state
                state_preprocessed = next_state_preprocessed
                timestep += 1

                # Train agent after collecting sufficient data
                if timestep >= start_timesteps and replay_buffer.size() > batch_size:
                    agent.train(replay_buffer, batch_size)

                if done or ep_timesteps >= max_timesteps:
                    break

            episode += 1
            episode_rewards.append(ep_reward)
            episode_timesteps.append(ep_timesteps)

            # Logging
            if episode % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-log_interval:])
                avg_timesteps = np.mean(episode_timesteps[-log_interval:])

                print(f"Episode {episode}\tTimestep {timestep}")
                print(f"Average Reward: {avg_reward:.2f}\tAverage Timesteps: {avg_timesteps:.2f}")
                logger.info(f"Episode {episode}\tTimestep {timestep}")
                logger.info(f"Average Reward: {avg_reward:.2f}\tAverage Timesteps: {avg_timesteps:.2f}")

                # Save the model if it's the best so far
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    # Save the best model
                    torch.save(agent.actor.state_dict(), 'ddpg_best_actor.pth')
                    torch.save(agent.critic.state_dict(), 'ddpg_best_critic.pth')
                    print(f"New best average reward: {best_avg_reward:.2f}, model saved.")
                    logger.info(f"New best average reward: {best_avg_reward:.2f}, model saved.")

                # Log to TensorBoard
                writer.add_scalar('Average Reward', avg_reward, episode)
                writer.add_scalar('Average Timesteps', avg_timesteps, episode)

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        logger.info("Training interrupted by user.")

    except Exception as e:
        print(f"An error occurred during training at episode {episode}, timestep {timestep}: {e}")
        logger.error(f"An error occurred during training at episode {episode}, timestep {timestep}: {e}")
        traceback.print_exc()

    finally:
        # Save the final model
        torch.save(agent.actor.state_dict(), "ddpg_actor.pth")
        torch.save(agent.critic.state_dict(), "ddpg_critic.pth")
        print("Final models saved.")
        logger.info("Final models saved.")

        # Close the environment
        env.close()
        logger.info("Environment closed.")

        # Terminate the simulator
        terminate_simulator(simulator_process)
        logger.info("Simulator terminated.")

        # Close TensorBoard writer
        writer.close()
        logger.info("TensorBoard writer closed.")

        # Plotting the training results
        plt.figure()
        plt.plot(range(1, episode + 1), episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('DDPG Training: Total Reward per Episode')
        plt.savefig('ddpg_training_rewards.png')
        plt.show()

        # Optionally, plot the average reward over a moving window
        window_size = 10
        if len(episode_rewards) >= window_size:
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.figure()
            plt.plot(range(window_size, episode + 1), moving_avg)
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.title(f'DDPG Training: Moving Average Reward (Window Size {window_size})')
            plt.savefig('ddpg_training_moving_avg_rewards.png')
            plt.show()
