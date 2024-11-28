import gym
#import gymnasium as gym
import gym_donkeycar
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import argparse
import subprocess
import time

# Utility function to launch the simulator
def launch_simulator(sim_path, port, gui):
    try:
        args = [sim_path, "--port", str(port)]
        if not gui:
            args.extend(["-batchmode", "-nographics"])
        process = subprocess.Popen(args)
        time.sleep(10)  # Wait for the simulator to initialize
        return process
    except Exception as e:
        print(f"Error launching simulator: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", required=True, help="Path to simulator executable")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--env_name", type=str, default="donkey-generated-roads-v0")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--timesteps", type=int, default=100000)
    args = parser.parse_args()

    # Launch the simulator
    simulator_process = launch_simulator(args.sim, args.port, args.gui)

    # Create the environment
    env = gym.make(args.env_name)

    # Configure action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    if args.test:
        # Load the trained model and evaluate
        model = DDPG.load("ddpg_donkeycar")
        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()
    else:
        # Train the model
        model = DDPG(
            "MlpPolicy", 
            env, 
            action_noise=action_noise, 
            buffer_size=100000,
            verbose=1, 
            tensorboard_log="./ddpg_donkeycar_tensorboard/"
        )
        model.learn(total_timesteps=args.timesteps)
        model.save("ddpg_donkeycar")

    simulator_process.terminate()
