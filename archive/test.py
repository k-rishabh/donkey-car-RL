import os
import gym
import gym_donkeycar
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# SET UP ENVIRONMENT
# You can also launch the simulator separately
# in that case, you don't need to pass a `conf` object


#Change this SIM_PATH in .env file 
exe_path = os.getenv("SIM_PATH")
exe_path = os.path.abspath(exe_path)

port = 9091

conf = { "exe_path" : exe_path, "port" : port }

env = gym.make("donkey-generated-track-v0", conf=conf)

# PLAY
obs = env.reset()
for t in range(100):
  action = np.array([0.0, 0.5]) # drive straight with small speed
  # execute the action
  obs, reward, done, info = env.step(action)

# Exit the scene
env.close()
