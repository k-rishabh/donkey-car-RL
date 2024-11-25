import gym
import gym_donkeycar
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import cv2
import argparse

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Output range [-1, 1]
        )
        self.max_action = max_action

    def forward(self, state):
        return self.net(state) * self.max_action


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards).reshape(-1, 1),
            np.array(next_states),
            np.array(dones).reshape(-1, 1)
        )


# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, lr=1e-3):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        action += noise_scale * np.random.normal(size=action.shape)  # Add exploration noise
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Critic loss
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# Preprocess State
def preprocess_state(state):
    """Convert RGB state to grayscale, resize, normalize, and flatten."""
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return resized.flatten() / 255.0


# Training Loop
def train(env, agent, replay_buffer, episodes, max_timesteps, batch_size, noise_scale, start_timesteps):
    for episode in range(episodes):
        state = env.reset()
        state = preprocess_state(state)
        episode_reward = 0

        for t in range(max_timesteps):
            # Select action
            if len(replay_buffer.buffer) < start_timesteps:
                action = env.action_space.sample()  # Random exploration
            else:
                action = agent.select_action(state, noise_scale)

            # Step environment
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)

            # Store transition
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            # Train agent
            if len(replay_buffer.buffer) > batch_size:
                agent.train(replay_buffer, batch_size)

            if done:
                break

        print(f"Episode {episode + 1}: Reward = {episode_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, required=True, help="Path to the Donkey Car simulator executable.")
    parser.add_argument("--port", type=int, default=9091, help="Simulator port.")
    args = parser.parse_args()

    # Initialize environment
    env = gym.make("donkey-generated-roads-v0")

    # Hyperparameters
    state_dim = 84 * 84  # Preprocessed state size
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    episodes = 1000
    max_timesteps = 1000
    batch_size = 256
    noise_scale = 0.1
    start_timesteps = 10000

    # Initialize agent and replay buffer
    agent = DDPGAgent(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    # Train agent
    train(env, agent, replay_buffer, episodes, max_timesteps, batch_size, noise_scale, start_timesteps)

    env.close()
