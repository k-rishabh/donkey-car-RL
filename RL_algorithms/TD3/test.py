import os
import gym
import gym_donkeycar
import cv2
import numpy as np
import torch
from TD3 import TD3Agent  # Import your TD3Agent
from dotenv import load_dotenv
import argparse
import time

# Preprocess state
def preprocess_state(state):

    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    state = cv2.resize(state, (84, 84))  # Resize for dimensionality reduction
    cv2.imshow("Processed Frame", state)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
        cv2.destroyAllWindows()
        exit()
    state = state.flatten() / 255.0  # Flatten and normalize
    return state

# Load the TD3 agent's weights
def load_agent(agent, actor_path):
    
    agent.actor.load_state_dict(torch.load(actor_path))
    agent.actor.eval()
    print("Loaded trained actor model.")

# Testing loop
def test_td3(agent, env, num_episodes, max_timesteps):

    for episode in range(num_episodes):
        state = preprocess_state(env.reset())
        done = False
        episode_reward = 0

        for t in range(max_timesteps):
            # Select action without exploration noise
            action = agent.select_action(state)

            # Execute the action
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)
            state = next_state
            episode_reward += reward

            if done:
                break

        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="TD3 Testing Script")
    parser.add_argument("--sim_path", type=str, required=True, help="Path to the simulator executable.")
    parser.add_argument("--env_name", type=str, default="donkey-generated-track-v0", help="Donkey simulator environment name.")
    parser.add_argument("--port", type=int, default=9091, help="Port for the simulator.")
    parser.add_argument("--actor_path", type=str, required=True, help="Path to the saved actor model.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of testing episodes.")
    parser.add_argument("--timesteps", type=int, default=1000, help="Maximum timesteps per episode.")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Initialize environment
    conf = {"exe_path": args.sim_path, "port": args.port, "env_name": args.env_name}
    env = gym.make(args.env_name, conf=conf)

    # TD3 parameters
    state_dim = 84 * 84  # Must match training dimensions
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize the agent
    agent = TD3Agent(state_dim, action_dim, max_action)

    # Load trained weights
    load_agent(agent, args.actor_path)

    # Test the agent
    test_td3(agent, env, num_episodes=args.episodes, max_timesteps=args.timesteps)

    # Cleanup
    env.close()