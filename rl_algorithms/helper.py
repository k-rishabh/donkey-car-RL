
import subprocess
import time
import sys
import numpy as np


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
def scale_action(action, env):
    action = np.clip(action, -1, 1)
    action_low = env.action_space.low
    action_high = env.action_space.high
    scaled_action = action * (action_high - action_low) / 2 + (action_high + action_low) / 2
    return scaled_action
