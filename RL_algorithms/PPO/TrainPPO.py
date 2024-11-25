'''
file: TrainPPO.py
python TrainPPO.py --sim /path/donkey_sim.exe --env_name donkey-generated-roads-v0   #(FOR TRAINING)
python TrainPPO.py --sim /path/donkey_sim.exe --env_name donkey-generated-roads-v0 --test --gui      # (FOR TESTING GUI)
python TrainPPO.py --sim /path/donkey_sim.exe --env_name donkey-generated-roads-v0 --test     # (FOR TESTING HEADLESS) 
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
import torch.nn.functional as F
import numpy as np
import cv2 
from collections import deque
import traceback
import logging
from PPO.PPO import Memory, PPOAgent
from helper import *
from PPO import *

# Optional: For TensorBoard integration
from torch.utils.tensorboard import SummaryWriter

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        filename='training.log',  # Log file name
        filemode='a',              # Append mode
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger()

    # Optional: Initialize TensorBoard writer
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
    max_action = float(env.action_space.high[0])  # Assuming symmetric action space

    # Initialize the PPO agent
    agent = PPOAgent(state_dim, action_dim)

    if args.test:
        # Load the trained policy network
        agent.policy_old.load_state_dict(torch.load("ppo_donkey_policy.pth", map_location=device))
        print("Loaded trained policy network.")
        logger.info("Loaded trained policy network for testing.")

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
    max_episodes = 1000
    max_timesteps = 1000
    update_timestep = 4000  											
    log_interval = 1       										

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

                # Optional: Log to TensorBoard
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
