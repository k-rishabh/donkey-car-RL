import argparse
import os
import gym
import gym_donkeycar
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
from helper import *


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

# Critic Network (Twin Critic)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1 architecture
        self.fc1_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc1_out = nn.Linear(hidden_dim, 1)
        # Q2 architecture
        self.fc2_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_out = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)
        # Q1 forward
        x1 = F.relu(self.fc1_1(xu))
        x1 = F.relu(self.fc1_2(x1))
        x1 = self.fc1_out(x1)
        # Q2 forward
        x2 = F.relu(self.fc2_1(xu))
        x2 = F.relu(self.fc2_2(x2))
        x2 = self.fc2_out(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], dim=1)
        x1 = F.relu(self.fc1_1(xu))
        x1 = F.relu(self.fc1_2(x1))
        x1 = self.fc1_out(x1)
        return x1

# TD3 Agent
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005, 
                 policy_noise=0.1, noise_clip=0.5, policy_delay=2):
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
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_delay = policy_delay
        self.total_it = 0

        # For logging losses
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.q_value_history = []

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.gamma * target_Q)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Record critic loss
        self.critic_loss_history.append(critic_loss.item())

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Record actor loss
            self.actor_loss_history.append(actor_loss.item())

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # Record average Q-value
            avg_Q = current_Q1.mean().item()
            self.q_value_history.append(avg_Q)


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
        default="donkey-mountain-track-v0",
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
    state_dim = 84 * 84  # After preprocessing
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize the TD3 agent
    agent = TD3Agent(state_dim, action_dim, max_action)

    # Initialize the replay buffer
    replay_buffer = ReplayBuffer()

    # Training parameters
    max_episodes = 100
    max_timesteps = 100
    batch_size = 256
    exploration_noise = 0.1
    start_timesteps = 1000  # Number of timesteps to collect data with random policy
    log_interval = 1

    timestep = 0
    episode = 0

    # Initialize metrics
    episode_rewards = []
    episode_timesteps = []
    actor_losses = []
    critic_losses = []
    avg_q_values = []
    noise_magnitudes = []

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
                    # Record noise magnitude as maximum possible (since action is random)
                    noise_magnitude = exploration_noise * max_action
                else:
                    # Select action according to policy
                    action = agent.select_action(state_preprocessed)
                    # Add exploration noise
                    noise = np.random.normal(0, exploration_noise, size=action_dim)
                    action = (action + noise).clip(
                        env.action_space.low, env.action_space.high
                    )
                    # Record noise magnitude
                    noise_magnitude = np.linalg.norm(noise)

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
                    # Record losses and average Q-values
                    if len(agent.actor_loss_history) > 0:
                        actor_losses.append(agent.actor_loss_history[-1])
                        critic_losses.append(agent.critic_loss_history[-1])
                    if len(agent.q_value_history) > 0:
                        avg_q_values.append(agent.q_value_history[-1])

                # Record noise magnitude
                noise_magnitudes.append(noise_magnitude)

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

                # Log to TensorBoard
                writer.add_scalar('Average Reward', avg_reward, episode)
                writer.add_scalar('Average Timesteps', avg_timesteps, episode)

                # Also log losses and average Q-values if available
                if len(actor_losses) > 0:
                    writer.add_scalar('Actor Loss', actor_losses[-1], episode)
                    writer.add_scalar('Critic Loss', critic_losses[-1], episode)
                    writer.add_scalar('Average Q-Value', avg_q_values[-1], episode)

                # Log noise magnitude
                writer.add_scalar('Exploration Noise Magnitude', noise_magnitude, episode)

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        logger.info("Training interrupted by user.")

    except Exception as e:
        print(f"An error occurred during training at episode {episode}, timestep {timestep}: {e}")
        logger.error(f"An error occurred during training at episode {episode}, timestep {timestep}: {e}")
        traceback.print_exc()

    finally:
        # Save the final model
        torch.save(agent.actor.state_dict(), "td3_actor.pth")
        torch.save(agent.critic.state_dict(), "td3_critic.pth")
        print("Models saved.")
        logger.info("Models saved.")

        # Plot rewards vs episodes
        try:
            plt.figure()
            plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Total Reward vs Episode')
            plt.grid(True)
            plt.savefig('rewards_vs_episodes.png')
            plt.show()
            logger.info("Reward plot saved.")

            # Plot actor and critic losses
            plt.figure()
            plt.plot(range(1, len(critic_losses) + 1), critic_losses, label='Critic Loss')
            plt.plot(range(1, len(actor_losses) + 1), actor_losses, label='Actor Loss')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Actor and Critic Losses')
            plt.legend()
            plt.grid(True)
            plt.savefig('losses.png')
            plt.show()
            logger.info("Loss plots saved.")

            # Plot average Q-values
            plt.figure()
            plt.plot(range(1, len(avg_q_values) + 1), avg_q_values)
            plt.xlabel('Training Steps')
            plt.ylabel('Average Q-Value')
            plt.title('Average Q-Value Over Time')
            plt.grid(True)
            plt.savefig('average_q_values.png')
            plt.show()
            logger.info("Average Q-Value plot saved.")

            # Plot exploration noise magnitude
            plt.figure()
            plt.plot(range(1, len(noise_magnitudes) + 1), noise_magnitudes)
            plt.xlabel('Time Steps')
            plt.ylabel('Noise Magnitude')
            plt.title('Exploration Noise Magnitude Over Time')
            plt.grid(True)
            plt.savefig('exploration_noise.png')
            plt.show()
            logger.info("Exploration noise plot saved.")

        except Exception as e:
            print(f"Failed to plot metrics: {e}")
            logger.error(f"Failed to plot metrics: {e}")

        # Close the environment
        env.close()
        logger.info("Environment closed.")

        # Terminate the simulator
        terminate_simulator(simulator_process)
        logger.info("Simulator terminated.")

        # Close TensorBoard writer
        writer.close()
        logger.info("TensorBoard writer closed.")
