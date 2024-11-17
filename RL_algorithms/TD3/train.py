#created seperate script for training TD3 agent for now to adjust parameters to pass but we will need to train it by main Train.py 
import os
import gym
import gym_donkeycar
import numpy as np
from TD3 import TD3Agent, ReplayBuffer
#from utils import ReplayBuffer
from dotenv import load_dotenv
import torch
import cv2


print(torch.cuda.is_available())
print(torch.version.cuda)



load_dotenv()
exe_path = os.getenv("SIM_PATH")
exe_path = os.path.abspath(exe_path)

port = 9091
conf = {"exe_path": exe_path, "port": port}
env = gym.make("donkey-generated-track-v0", conf=conf)

# TD3 Parameters
state_dim = 84 * 84  # Grayscale flattened image size
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
agent = TD3Agent(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(state_dim, action_dim,max_size=1e5)


def preprocess_state(state):
    
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    state = cv2.resize(state, (84, 84)) 
    state = state.flatten() / 255.0  # Normalize to [0, 1]
    return state


# Training loop
num_episodes = 100
batch_size = 100
for episode in range(num_episodes):
    state = env.reset()
    state = preprocess_state(state)
    done = False
    episode_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(next_state)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if replay_buffer.size > batch_size:
            agent.train(replay_buffer, batch_size)

    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")


torch.save(agent.actor.state_dict(), "td3_actor.pth")
torch.save(agent.critic.state_dict(), "td3_critic.pth")
print("Models saved successfully!")

env.close()
