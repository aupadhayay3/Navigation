# Imports
import numpy as np
import random
from collections import namedtuple, deque

from model import DeepQNetwork 

import torch
import torch.nn as nn
import torch.nn.functional as F
 


# Hyperparameters
GAMMA_VALUE = 0.99
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
UPDATE_EVERY = 4
TAU = 1e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, state_size, action_size, seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
   
        
        self.predicted_Qnetwork = DeepQNetwork(self.state_size, self.action_size, self.seed).to(device)
        self.target_Qnetwork = DeepQNetwork(self.state_size, self.action_size, self.seed).to(device)
        
        self.optimizer = torch.optim.Adam(self.predicted_Qnetwork.parameters(), lr=LEARNING_RATE)
        
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        
        self.t = 0
        
        
    # agent produces an action
    def act(self, state, epsilon):
        
        # creates tensor from numpy array
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.predicted_Qnetwork.eval()
          
        with torch.no_grad():
            action_values = self.predicted_Qnetwork.forward(state).squeeze()
            
        rand = random.random()
        
        action = None
        # select random choice with probability epsilon
        if rand < epsilon:
            action = random.choice(range(self.action_size))
        else:
            # obtain the max Q values and indices
            # then get the index of it to obtain the action 
            action = action_values.max(0)[1].squeeze().item()
    
        
        self.predicted_Qnetwork.train()
        return action
    
    
    # adds experiences to the replay buffer and learns from a minibatch of experiences
    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        self.t = (self.t + 1) % UPDATE_EVERY
        # ensures there are enough memories to take a batch of memories
        if len(self.replay_buffer) > BATCH_SIZE:
            if self.t == 0:
                # get random batch of memories from the replay buffer
                sampled_experiences = self.replay_buffer.sample()
                # learn from the batch of memories
                self.learn(sampled_experiences)
        
        
   
    # update the deep Q network based on experiences
    def learn(self, experiences, gamma=GAMMA_VALUE):
        
        # unpack experiences
        states, actions, rewards, next_states, dones = experiences
        
        # TO DO: use states and actions to produce the q value from the current network
        action_values = self.predicted_Qnetwork.forward(states)
        predictedQ = action_values.gather(1, actions).squeeze()
        
        # TO DO: use rewards and next_states to produce the q value from the target network
        a = self.target_Qnetwork.forward(next_states).detach()        
        target_action_values = self.target_Qnetwork.forward(next_states).detach().max(1)[0]
        targetQ = rewards.squeeze() + gamma*target_action_values.squeeze()*((1-dones).squeeze())
        
        
        # TO DO: use gradient descent to update the network 
        self.optimizer.zero_grad()
        loss = F.mse_loss(predictedQ, targetQ)
        loss.backward()
        self.optimizer.step()
        
        
        # soft update target network  
        self.update_target(self.predicted_Qnetwork, self.target_Qnetwork, TAU)
        
        
    # adjust target network to be closer to recent q network
    def update_target(self, predictedQnetwork, targetQnetwork, tau):
 
        for pParam, tParam in zip(predictedQnetwork.parameters(), targetQnetwork.parameters()):
            # in place copying 
            tParam.data.copy_(tau*pParam.data + (1.0-tau)*tParam.data)
            
        
        
    
    
    
    
# NEED TO KNOW WHAT DATA STRUCTURE TO USE FOR MEMORY 
class ReplayBuffer():
    
    def __init__(self, buffer_size, batch_size, seed):
        
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    
    # returns minibatch of experiences from memory
    def sample(self):
        
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done*1 for e in experiences if e is not None])).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    # adds an experience tuple to the memory data structure
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    # returns the current size of internal memory.
    def __len__(self):
        return len(self.memory)
    
    

    
    
    

    