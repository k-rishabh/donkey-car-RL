# ppo_train.py

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
import cv2
import logging
from collections import deque
import traceback
from torch.utils.tensorboard import SummaryWriter

# Import matplotlib for plotting
import matplotlib.pyplot as plt

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Policy Network for Continuous Actions
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.mean.weight.data.mul_(0.1)
        self.mean.bias.data.mul_(0.0)
        self.log_std.weight.data.mul_(0.1)
        self.log_std.bias.data.mul_(0.0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Stability
        
        std = torch.exp(log_std)
        return mean, std

# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        state_value = self.value_head(x)
        return state_value

# Define the Memory Class
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []           # Pre-tanh actions
        self.actions_tanh = []      # Tanh-transformed actions
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = []
        self.returns = []
    
    def clear(self):
        self.states = []
        self.actions = []
        self.actions_tanh = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = []
        self.returns = []

# Define the PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95, clip_epsilon=0.2, 
                 K_epochs=4, minibatch_size=64, entropy_coef=0.01):
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.K_epochs = K_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef

        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.policy_old = PolicyNetwork(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.value_net = ValueNetwork(state_dim).to(device)
        
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_net.parameters(), 'lr': lr}
        ])
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        mean, std = self.policy_old(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()  # Allows gradient flow
        action_tanh = torch.tanh(action)
        # Compute log probability with correction
        logprob = dist.log_prob(action) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        logprob = logprob.sum(dim=-1)
        action_tanh = action_tanh.detach().cpu().numpy()
        logprob = logprob.detach().cpu().numpy()
        action = action.detach().cpu().numpy()  # Pre-tanh action
        return action_tanh, logprob, action  # Return tanh action, logprob, and pre-tanh action
    
    def compute_gae(self, rewards, dones, values):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        advantages = np.array(returns) - values[:-1]
        return returns, advantages
    
    def update(self, memory):
        # Convert lists to tensors
        old_states = torch.FloatTensor(np.array(memory.states)).to(device)
        old_actions = torch.FloatTensor(np.array(memory.actions)).to(device)        # Pre-tanh actions
        old_logprobs = torch.FloatTensor(np.array(memory.logprobs)).to(device)
        old_returns = torch.FloatTensor(np.array(memory.returns)).to(device)
        old_advantages = torch.FloatTensor(np.array(memory.advantages)).to(device)
        
        # Normalize advantages
        old_advantages = (old_advantages - old_advantages.mean()) / (old_advantages.std() + 1e-8)
        
        dataset = torch.utils.data.TensorDataset(old_states, old_actions, old_logprobs, old_returns, old_advantages)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)
        
        for _ in range(self.K_epochs):
            for batch in dataloader:
                states, actions, logprobs, returns, advantages = batch
                
                # Evaluating old actions and values
                mean, std = self.policy(states)
                dist = torch.distributions.Normal(mean, std)
                new_logprobs = dist.log_prob(actions) - torch.log(1 - torch.tanh(actions).pow(2) + 1e-6)
                new_logprobs = new_logprobs.sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Finding the ratio (pi_theta / pi_theta_old)
                ratios = torch.exp(new_logprobs - logprobs)
                
                # Finding Surrogate Loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Finding Value Loss
                state_values = self.value_net(states).squeeze()
                value_loss = self.MseLoss(state_values, returns)
                
                # Total Loss
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                
                # Take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

# Function to launch the simulator
def launch_simulator(sim_path, port, gui=False):
    try:
        if gui:
            print(f"Launching simulator from {sim_path} on port {port} with GUI...")
            # Launch simulator without headless flags
            simulator_process = subprocess.Popen([
                sim_path,
                "--port", str(port)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            print(f"Launching simulator from {sim_path} on port {port} in headless mode...")
            # Launch simulator with headless flags
            simulator_process = subprocess.Popen([
                sim_path,
                "--port", str(port),
                "-batchmode",
                "-nographics",
                "-silent-crashes"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
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

    # Setup logging
    logging.basicConfig(
        filename='training.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger()

    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/ppo_training')

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

    parser = argparse.ArgumentParser(description="ppo_train")
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

    # Initialize the PPO agent
    agent = PPOAgent(state_dim, action_dim)

    if args.test:
        # Load the trained policy network
        agent.policy_old.load_state_dict(torch.load("ppo_best_policy.pth", map_location=device))
        print("Loaded best-performing policy network.")
        logger.info("Loaded best-performing policy network for testing.")

        # Run the testing loop
        try:
            while True:
                state = env.reset()
                done = False
                ep_reward = 0
                ep_timesteps = 0

                while not done:
                    # Preprocess state
                    state_preprocessed = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
                    state_preprocessed = cv2.resize(state_preprocessed, (84, 84))
                    state_preprocessed = state_preprocessed.flatten() / 255.0

                    # Select action without exploration
                    with torch.no_grad():
                        action_tanh, _, _ = agent.select_action(state_preprocessed)
                    # Scale action to environment's action space
                    scaled_action = scale_action(action_tanh)
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

    # Initialize memory
    memory = Memory()

    # Training parameters
    max_episodes = 4000
    max_timesteps = 1000
    update_timestep = 4000
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
            for t in range(max_timesteps):
                # Preprocess state
                state_preprocessed = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
                state_preprocessed = cv2.resize(state_preprocessed, (84, 84))
                state_preprocessed = state_preprocessed.flatten() / 255.0

                # Select action with exploration
                action_tanh, logprob, action = agent.select_action(state_preprocessed)
                # Scale action to environment's action space
                scaled_action = scale_action(action_tanh)
                scaled_action = scaled_action.clip(env.action_space.low, env.action_space.high)

                next_state, reward, done, _ = env.step(scaled_action)

                # Save in memory
                memory.states.append(state_preprocessed)
                memory.actions.append(action)            # Pre-tanh action
                memory.actions_tanh.append(action_tanh)  # Tanh-transformed action
                memory.logprobs.append(logprob)
                memory.rewards.append(reward)
                memory.dones.append(done)

                state = next_state
                ep_reward += reward
                timestep += 1
                ep_timesteps += 1

                # Update PPO Agent
                if timestep % update_timestep == 0:
                    # Compute advantages
                    # Preprocess the last state for value estimation
                    state_preprocessed = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
                    state_preprocessed = cv2.resize(state_preprocessed, (84, 84))
                    state_preprocessed = state_preprocessed.flatten() / 255.0
                    with torch.no_grad():
                        next_state_tensor = torch.FloatTensor(state_preprocessed).to(device)
                        next_value = agent.value_net(next_state_tensor).item()

                    # Compute values for all states in memory
                    states_tensor = torch.FloatTensor(np.array(memory.states)).to(device)
                    values = agent.value_net(states_tensor).detach().cpu().numpy().flatten()

                    # Append next_value
                    values = np.append(values, next_value)

                    returns, advantages = agent.compute_gae(memory.rewards, memory.dones, values)
                    memory.returns = returns
                    memory.advantages = advantages

                    # Update the policy
                    agent.update(memory)

                    # Clear memory
                    memory.clear()

                if done:
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
                    torch.save(agent.policy.state_dict(), 'ppo_best_policy.pth')
                    torch.save(agent.value_net.state_dict(), 'ppo_best_value.pth')
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
        torch.save(agent.policy.state_dict(), "ppo_donkey_policy.pth")
        torch.save(agent.value_net.state_dict(), "ppo_donkey_value.pth")
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
        plt.title('PPO Training: Total Reward per Episode')
        plt.savefig('ppo_training_rewards.png')
        plt.show()

        # Optionally, plot the average reward over a moving window
        window_size = 10
        if len(episode_rewards) >= window_size:
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.figure()
            plt.plot(range(window_size, episode + 1), moving_avg)
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.title(f'PPO Training: Moving Average Reward (Window Size {window_size})')
            plt.savefig('ppo_training_moving_avg_rewards.png')
            plt.show()
