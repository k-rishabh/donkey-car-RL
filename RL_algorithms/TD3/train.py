# Created separate script for training TD3 agent for now to adjust parameters to pass
import os
import gym
import gym_donkeycar
import numpy as np
from TD3 import TD3Agent, ReplayBuffer
from dotenv import load_dotenv
import torch
import cv2
import subprocess
import time


print(torch.cuda.is_available())
print(torch.version.cuda)

# Load environment variables
load_dotenv()
exe_path = os.getenv("SIM_PATH")
exe_path = os.path.abspath(exe_path)

# Configure the environment for headless mode
port = 9091
# conf = {"exe_path": exe_path, "port": port,"headless":False}  
# env = gym.make("donkey-generated-roads-v0", conf=conf)

simulator_process = subprocess.Popen([
                exe_path,
                "--port", str(port),
                "-batchmode",
                "-nographics",
                "-silent-crashes"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

time.sleep(10)
conf = {"exe_path": exe_path, "port": port,"headless":True}
env = gym.make("donkey-generated-roads-v0", conf=conf)

# TD3 Parameters
state_dim = 84 * 84  # Grayscale flattened image size
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
agent = TD3Agent(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=1e5)

def preprocess_state(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    state = cv2.resize(state, (84, 84)) 
    state = state.flatten() / 255.0  # Normalize to [0, 1]
    return state

# Training loop
num_episodes = 500
batch_size = 100
warmup_steps = 10000  # Random exploration phase
noise_std = 0.1  # Exploration noise

for episode in range(num_episodes):
    state = env.reset()
    state = preprocess_state(state)
    done = False
    episode_reward = 0

    while not done:
        if replay_buffer.size < warmup_steps:
            action = env.action_space.sample()  # Random action during warm-up
        else:
            action = agent.select_action(state)
            # Add exploration noise
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = np.clip(action + noise, env.action_space.low, env.action_space.high)

        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(next_state)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if replay_buffer.size > batch_size:
            agent.train(replay_buffer, batch_size)

    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

# Save models after training
torch.save(agent.actor.state_dict(), "td3_actor.pth")
torch.save(agent.critic.state_dict(), "td3_critic.pth")
print("Models saved successfully!")

env.close()

# Created separate script for training TD3 agent for now to adjust parameters to pass
