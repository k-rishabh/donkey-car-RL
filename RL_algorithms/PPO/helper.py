
import subprocess
import time
import sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque


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
