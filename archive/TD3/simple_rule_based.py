import os
import gym
import gym_donkeycar
import numpy as np
import cv2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
exe_path = os.getenv("SIM_PATH")  # Path to simulator executable
exe_path = os.path.abspath(exe_path)
port = 9091

# Configuration for the Donkey Simulator
conf = {"exe_path": exe_path, "port": port, "camera": "FPP"}


# Preprocess the image
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (84, 84))            # Resize for simplicity
    edges = cv2.Canny(resized, 50, 150)             # Apply edge detection
    return edges


# Calculate steering based on processed frame
def calculate_steering(processed_frame):
    height, width = processed_frame.shape
    middle = width // 2

    # Look at the bottom section of the image for road edges
    road_indices = np.where(processed_frame[int(height * 0.7)] > 0)[0]  # Bottom 20% of the frame

    if len(road_indices) > 0:
        road_center = (road_indices[0] + road_indices[-1]) // 2  # Midpoint of road edges
        deviation = road_center - middle
        steering = -deviation / middle  # Normalize deviation to [-1, 1]
    else:
        # If no road edges are detected, default to driving straight
        steering = 0
        throttle = 0

    return steering


# Control car in the simulator
def control_car(frame, env):

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Visualize the processed frame in an OpenCV window
    cv2.imshow("Processed Frame", processed_frame)

    # Press 'q' to quit the visualization and simulation
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return frame, True  # Exit simulation when 'q' is pressed

    # Calculate steering
    steering = calculate_steering(processed_frame)
    throttle = 0.01  # Set a constant speed

    # Action vector: [steering, throttle]
    action = np.array([steering, throttle])
    next_frame, reward, done, info = env.step(action)

    return next_frame, done


# Main function to run the rule-based driving system
def main():
    # Initialize the Donkey Simulator environment
    env = gym.make("donkey-generated-track-v0", conf=conf)
    obs = env.reset()
    # print(obs)

    print("Starting the simulation...")

    try:
        # Main control loop
        done = False
        while not done:
            obs, done = control_car(obs, env)
            # print(obs.shape)

    except KeyboardInterrupt:
        print("Simulation stopped by user.")

    finally:
        # Clean up
        env.close()
        cv2.destroyAllWindows()  # Close OpenCV windows
        print("Simulation ended.")

if __name__ == "__main__":
    main()
