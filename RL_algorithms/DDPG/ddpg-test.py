import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from collections import deque
import random
import gym
import gym_donkeycar

class DDPGAgent:
    def __init__(self, state_shape=(120, 160, 3), action_dim=2,
                 actor_lr=0.0001, critic_lr=0.001,
                 gamma=0.99, tau=0.005, buffer_size=50000):
        
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.gamma = gamma  # Discount factor
        self.tau = tau     # Target network update rate
        
        # Action bounds for DonkeyCar (steering, throttle)
        self.action_low = np.array([-1.0, 0.0])
        self.action_high = np.array([1.0, 1.0])
        
        # Initialize networks
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()
        
        # Copy weights
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        # Experience replay buffer
        self.buffer = deque(maxlen=buffer_size)
        
        # Noise process for exploration
        self.noise = OUNoise(action_dim)
        
    def build_actor(self):
        """Actor network for DDPG"""
        inputs = layers.Input(shape=self.state_shape)
        
        # Normalize input
        x = layers.Lambda(lambda x: x/255.0)(inputs)
        
        # Convolutional layers
        x = layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
        x = layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
        x = layers.Conv2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        
        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        
        # Output layer with tanh activation for bounded actions
        outputs = layers.Dense(self.action_dim, activation='tanh')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.0001))
        return model
        
    def build_critic(self):
        """Critic network for DDPG"""
        # State input
        state_input = layers.Input(shape=self.state_shape)
        x = layers.Lambda(lambda x: x/255.0)(state_input)
        
        # Convolutional layers for state
        x = layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
        x = layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
        x = layers.Conv2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.Flatten()(x)
        
        # Action input
        action_input = layers.Input(shape=(self.action_dim,))
        
        # Combine state and action pathways
        x = layers.Concatenate()([x, action_input])
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(1)(x)
        
        model = Model(inputs=[state_input, action_input], outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def get_action(self, state, add_noise=True):
        """Get action from actor network with optional exploration noise"""
        state = np.expand_dims(state, axis=0)
        action = self.actor.predict(state)[0]
        
        if add_noise:
            action += self.noise.sample()
        
        # Clip action to bounds and scale to environment range
        action = np.clip(action, -1, 1)
        scaled_action = self.scale_action(action)
        return scaled_action
    
    def scale_action(self, action):
        """Scale actions from [-1, 1] to environment range"""
        return (action + 1.0) * 0.5 * (self.action_high - self.action_low) + self.action_low
    
    def train(self, batch_size=64):
        """Train actor and critic networks"""
        if len(self.buffer) < batch_size:
            return
        
        # Sample random batch from buffer
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        
        # Get target actions from target actor
        target_actions = self.target_actor.predict(next_states)
        
        # Get target Q-values from target critic
        target_q_values = self.target_critic.predict([next_states, target_actions])
        
        # Compute target values (Bellman equation)
        target_values = rewards + self.gamma * target_q_values * (1 - dones)
        
        # Train critic
        self.critic.train_on_batch([states, actions], target_values)
        
        # Train actor using critic's gradient
        with tf.GradientTape() as tape:
            actor_actions = self.actor(states)
            critic_values = self.critic([states, actor_actions])
            actor_loss = -tf.math.reduce_mean(critic_values)
            
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Update target networks
        self.update_target_networks()
    
    def update_target_networks(self):
        """Soft update of target networks"""
        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        
        for i in range(len(actor_weights)):
            target_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * target_actor_weights[i]
        
        for i in range(len(critic_weights)):
            target_critic_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * target_critic_weights[i]
            
        self.target_actor.set_weights(target_actor_weights)
        self.target_critic.set_weights(target_critic_weights)

class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise"""
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.size) * self.mu
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

def train_donkey_ddpg(env, episodes=1000, batch_size=64, save_freq=100):
    """Training loop for DonkeyCar with DDPG"""
    agent = DDPGAgent()
    
    for episode in range(episodes):
        state, _, _, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get action from agent
            action = agent.get_action(state)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            agent.train(batch_size)
            
            episode_reward += reward
            state = next_state
            
        print(f"Episode: {episode}, Reward: {episode_reward}")
        
        # Save model periodically
        if episode % save_freq == 0:
            agent.actor.save(f'ddpg_actor_episode_{episode}.h5')
            agent.critic.save(f'ddpg_critic_episode_{episode}.h5')

import numpy as np
import tensorflow as tf
import os
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_training_logs():
    """Create directories and files for logging training progress"""
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("training_logs", current_time)
    model_dir = os.path.join(log_dir, "models")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    return log_dir, model_dir

def plot_training_results(episodes, rewards, steps, save_path):
    """Plot and save training metrics"""
    plt.figure(figsize=(12, 4))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot steps
    plt.subplot(1, 2, 2)
    plt.plot(episodes, steps)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_metrics.png'))
    plt.close()

def main():
    # Initialize environment and agent
    env = gym.make("donkey-generated-roads-v0")
    agent = DDPGAgent()
    
    # Training parameters
    MAX_EPISODES = 1000
    MAX_STEPS = 2000
    BATCH_SIZE = 64
    SAVE_FREQUENCY = 50
    INITIAL_EXPLORATION_EPISODES = 5
    
    # Create logging directories
    log_dir, model_dir = create_training_logs()
    
    # Training metrics
    episode_rewards = []
    episode_steps = []
    best_reward = float('-inf')
    
    try:
        for episode in range(MAX_EPISODES):
            state= env.reset()
            episode_reward = 0
            episode_step = 0
            
            # Reset noise process at the start of each episode
            agent.noise.reset()
            
            # Training progress bar
            pbar = tqdm(total=MAX_STEPS, desc=f'Episode {episode}')
            
            while True:
                # Initial random exploration
                if episode < INITIAL_EXPLORATION_EPISODES:
                    action = np.random.uniform(agent.action_low, agent.action_high)
                else:
                    action = agent.get_action(state, add_noise=True)
                
                # Take action in environment
                next_state, reward, done, _ = env.step(action)
                
                # Store experience in replay buffer
                agent.remember(state, action, reward, next_state, done)
                
                # Train agent if enough samples are available
                if len(agent.buffer) > BATCH_SIZE:
                    agent.train(BATCH_SIZE)
                
                episode_reward += reward
                episode_step += 1
                state = next_state
                
                pbar.update(1)
                
                if done or episode_step >= MAX_STEPS:
                    pbar.close()
                    break
            
            # Store episode metrics
            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)
            
            # Print episode summary
            print(f"\nEpisode: {episode}")
            print(f"Total Reward: {episode_reward:.2f}")
            print(f"Steps: {episode_step}")
            print(f"Buffer Size: {len(agent.buffer)}")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.actor.save(os.path.join(model_dir, 'best_actor.h5'))
                agent.critic.save(os.path.join(model_dir, 'best_critic.h5'))
            
            # Periodic model saving
            if episode % SAVE_FREQUENCY == 0:
                agent.actor.save(os.path.join(model_dir, f'actor_episode_{episode}.h5'))
                agent.critic.save(os.path.join(model_dir, f'critic_episode_{episode}.h5'))
                
                # Save training metrics
                np.save(os.path.join(log_dir, 'episode_rewards.npy'), episode_rewards)
                np.save(os.path.join(log_dir, 'episode_steps.npy'), episode_steps)
                
                # Plot training progress
                plot_training_results(
                    range(episode + 1),
                    episode_rewards,
                    episode_steps,
                    log_dir
                )
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        # Final save
        print("Saving final models...")
        agent.actor.save(os.path.join(model_dir, 'final_actor.h5'))
        agent.critic.save(os.path.join(model_dir, 'final_critic.h5'))
        
        # Save final metrics
        np.save(os.path.join(log_dir, 'episode_rewards.npy'), episode_rewards)
        np.save(os.path.join(log_dir, 'episode_steps.npy'), episode_steps)
        
        # Plot final training progress
        plot_training_results(
            range(len(episode_rewards)),
            episode_rewards,
            episode_steps,
            log_dir
        )
        
        env.close()
        print("Training finished.")

if __name__ == '__main__':
    main()