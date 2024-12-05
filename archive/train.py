#Trying to make a single file for training multiple models 
import os
import gym
import gym_donkeycar
import numpy as np
#from TD3.agent import Agent  #Need to change this according to model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Path to the simulator executable
exe_path = os.getenv("SIM_PATH")
exe_path = os.path.abspath(exe_path)

port = 9091

conf = { "exe_path" : exe_path, "port" : port }

env = gym.make("donkey-generated-track-v0", conf=conf)

# Training settings
num_episodes = 100
state_dim = 84 * 84 # 84x84 pixel image input (We might need to change this)
action_dim = 2  #angle and throttle
max_action = 1.0  

# Initialize agent
agent = Agent(state_dim, action_dim, max_action)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.select_action(state)  #We might need to handel pre-processing images here
        next_state, reward, done, info = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        agent.train()

    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
    
# Clean up the environment
env.close()
model_dir = "saved_models"


if not os.path.exists(model_dir):
    os.makedirs(model_dir)
save_path = os.path.join(model_dir, "final_agent_model.pth")
agent.save(save_path)