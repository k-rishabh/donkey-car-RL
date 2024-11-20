'''
file: ddpg_train.py
notes: DDPG implementation for continuous action spaces.
       Includes automatic simulator launch with optional GUI.
Usage:
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
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import logging
import traceback

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
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.max_action
        return x


# Critic Network (Q-function)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, u):
        x = torch.relu(self.fc1(torch.cat([x, u], dim=1)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))

        return (
            torch.FloatTensor(batch_states).to(device),
            torch.FloatTensor(batch_next_states).to(device),
            torch.FloatTensor(batch_actions).to(device),
            torch.FloatTensor(batch_rewards).unsqueeze(1).to(device),
            torch.FloatTensor(batch_dones).unsqueeze(1).to(device),
        )


# DDPG Agent
class DDPGAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        lr=1e-3,
        gamma=0.99,
        tau=0.005,
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self, replay_buffer, batch_size=256):
        # Sample replay buffer
        (
            state,
            next_state,
            action,
            reward,
            done,
        ) = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (1 - done) * self.gamma * target_Q

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q.detach())

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

        # Update the frozen target models
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
                [
                    sim_path,
                    "--port",
                    str(port),
                    "-batchmode",
                    "-nographics",
                    "-silent-crashes",
                ],
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


# Function to scale actions
def scale_action(action):
    action = np.clip(action, -1, 1)
    action_low = env.action_space.low
    action_high = env.action_space.high
    scaled_action = action * (action_high - action_low) / 2 + (action_high + action_low) / 2
    return scaled_action


# Main function
if __name__ == "__main__":

    # Optional: Setup logging
    logging.basicConfig(
        filename='ddpg_training.log',  # Log file name
        filemode='a',                  # Append mode
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger()

    # Optional: Initialize TensorBoard writer
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

    # Set environment variables for gym-donkeycar
    os.environ['DONKEY_SIM_PATH'] = args.sim
    os.environ['DONKEY_SIM_PORT'] = str(args.port)

    # Launch the simulator with or without GUI based on the --gui flag
    simulator_process = launch_simulator(args.sim, args.port, gui=args.gui)

    # Initialize the environment
    env = gym.make(env_id)
    state_dim = 84 * 84  # After preprocessing, the state is a flattened (84,84) image
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])  # Assuming symmetric action space

    # Initialize the DDPG agent
    agent = DDPGAgent(state_dim, action_dim, max_action)

    if args.test:
        # Load the trained policy network
        agent.actor.load_state_dict(torch.load("ddpg_actor.pth", map_location=device))
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
    max_timesteps = 1000
    exploration_noise = 0.1
    batch_size = 256
    start_timesteps = 10000  # Number of timesteps to collect transitions without training
    eval_freq = 5000         # Evaluate every eval_freq timesteps
    save_models = True
    total_timesteps = 0

    # Initialize metrics
    episode_rewards = []
    episode_timesteps = []

    # OUNoise for exploration
    class OUNoise:
        def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
            self.action_dim = action_dim
            self.mu = mu
            self.theta = theta
            self.sigma = sigma
            self.state = np.ones(self.action_dim) * self.mu

        def reset(self):
            self.state = np.ones(self.action_dim) * self.mu

        def sample(self):
            x = self.state
            dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
            self.state = x + dx
            return self.state

    ou_noise = OUNoise(action_dim)

    try:
        for episode in range(1, max_episodes + 1):
            state = env.reset()
            done = False
            ep_reward = 0
            ep_timesteps = 0

            ou_noise.reset()

            for t in range(max_timesteps):
                total_timesteps += 1

                # Preprocess state
                state_preprocessed = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
                state_preprocessed = cv2.resize(state_preprocessed, (84, 84))
                state_preprocessed = state_preprocessed.flatten() / 255.0

                # Select action with exploration noise
                if total_timesteps < start_timesteps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state_preprocessed)
                    noise = ou_noise.sample()
                    action = (action + noise).clip(-max_action, max_action)

                # Scale action to environment's action space
                scaled_action = scale_action(action)
                scaled_action = scaled_action.clip(env.action_space.low, env.action_space.high)

                next_state, reward, done, _ = env.step(scaled_action)

                # Preprocess next state
                next_state_preprocessed = cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY)
                next_state_preprocessed = cv2.resize(next_state_preprocessed, (84, 84))
                next_state_preprocessed = next_state_preprocessed.flatten() / 255.0

                # Store data in replay buffer
                replay_buffer.add(
                    (state_preprocessed, next_state_preprocessed, action, reward, float(done))
                )

                state = next_state
                ep_reward += reward
                ep_timesteps += 1

                if total_timesteps >= start_timesteps:
                    # Train agent
                    agent.train(replay_buffer, batch_size)

                if done:
                    break

            episode_rewards.append(ep_reward)
            episode_timesteps.append(ep_timesteps)

            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_timesteps = np.mean(episode_timesteps[-10:])

                print(f"Episode {episode}\tTotal Timesteps {total_timesteps}")
                print(f"Average Reward: {avg_reward:.2f}\tAverage Timesteps: {avg_timesteps:.2f}")
                logger.info(f"Episode {episode}\tTotal Timesteps {total_timesteps}")
                logger.info(f"Average Reward: {avg_reward:.2f}\tAverage Timesteps: {avg_timesteps:.2f}")

                # Optional: Log to TensorBoard
                writer.add_scalar('Average Reward', avg_reward, episode)
                writer.add_scalar('Average Timesteps', avg_timesteps, episode)

            # Save the model
            if save_models and episode % 100 == 0:
                torch.save(agent.actor.state_dict(), "ddpg_actor.pth")
                torch.save(agent.critic.state_dict(), "ddpg_critic.pth")
                print("Models saved.")
                logger.info("Models saved.")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        logger.info("Training interrupted by user.")

    except Exception as e:
        print(f"An error occurred during training at episode {episode}, timestep {total_timesteps}: {e}")
        logger.error(f"An error occurred during training at episode {episode}, timestep {total_timesteps}: {e}")
        traceback.print_exc()

    finally:
        # Save the final model
        torch.save(agent.actor.state_dict(), "ddpg_actor_final.pth")
        torch.save(agent.critic.state_dict(), "ddpg_critic_final.pth")
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
