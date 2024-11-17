# Main Things to do 
# 1.  Actor
# 2.  Critic
# 3. Target Network
#4. Writting train function to use from Train.py


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#Making this simple FFN for now
#This will predict action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):  #(mostly our state dim will be image 256*256 or 84*84 if resized)
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)         
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)    
        self.max_action = max_action    
    
    
    def forward(self, state):
        x = torch.relu(self.l1(state))
        x = torch.relu(self.l2(x))
        #print("max action = ", self.max_action)
        x = self.max_action * torch.tanh(self.l3(x))      #steering angle will be between -1 to 1 so trying with tanh
        #print("x = ", x)
        return x

#this will predict Q value 
#We will need to define two critc networks to predict two Q values Q1 and Q2   
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        
        
        
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)

        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def q1(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        return self.l3(q1)
    # we need to write this individually since for actor we need to paass only q1
    
class TD3Agent:
    #We will combine both actor and critic here
    def __init__(self,state_dim,action_dim,max_action,lr=0.05):
        self.actor = Actor(state_dim, action_dim, max_action).to("cuda")
        self.actor_target = Actor(state_dim, action_dim, max_action).to("cuda")
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim).to("cuda")
        self.critic_target = Critic(state_dim, action_dim).to("cuda")
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.max_action = max_action
        
    def select_action(self, state,noise_std=0.1):
        #print("state = ", state.shape)
        state = torch.Tensor(state.reshape(1, -1)).to("cuda")
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = (action + noise).clip(-self.max_action, self.max_action)
        return action
    
    def train(self, replay_buffer, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2): #setted this value from example documentation
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)

        state = torch.Tensor(state).to("cuda")
        action = torch.Tensor(action).to("cuda")
        
        with torch.no_grad():
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip) #adds noise to the action
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * discount * target_q
            
        # Get current Q-values
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()  #to update the weights of critics
        self.critic_optimizer.step()
        
        #updation of policy
        if replay_buffer.steps % 2 == 0:   #updating alternate steps
            # Compute actor loss
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)



#this is to store the earlier data
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1e6):
        
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.steps = 0 

        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.not_done[self.ptr] = 1.0 - done  # Convert done to not_done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.steps += 1
        

    def sample(self, batch_size):
        
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to("cuda"),
            torch.FloatTensor(self.action[ind]).to("cuda"),
            torch.FloatTensor(self.reward[ind]).to("cuda"),
            torch.FloatTensor(self.next_state[ind]).to("cuda"),
            torch.FloatTensor(self.not_done[ind]).to("cuda"),
        )
