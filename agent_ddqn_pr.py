# Imports
import numpy as np
import random
from collections import namedtuple, deque

from model import DeepQNetwork 
from agent import Agent
from agent import ReplayBuffer
from agent_ddqn import AgentDDQN


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
PRIORITY_CONSTANT = 1e-4
PRIORITY_A = 0.8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AgentDDQNPR(AgentDDQN):
    
    def __init__(self, state_size, action_size, seed):
        super().__init__(state_size, action_size, seed)
        self.replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
    
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

        
        torch_state = torch.from_numpy(state).float().to(device)
        torch_action = torch.tensor(action).long().to(device)
        torch_reward = torch.tensor(reward).float().to(device)
        torch_next_state = torch.from_numpy(next_state).float().to(device)
        torch_done = torch.tensor(1*done).float().to(device)

        
        pred = (self.predicted_Qnetwork.forward(torch_state).detach().squeeze()[torch_action].squeeze())        
        target_input_action = self.predicted_Qnetwork.forward(torch_next_state).detach().max(1)[1].unsqueeze(1)
        target_action_value = self.target_Qnetwork.forward(torch_next_state).detach().gather(1,target_input_action).squeeze()
        targ = torch_reward.squeeze() + GAMMA_VALUE*target_action_value.squeeze()*((1-torch_done).squeeze())
        priority = PRIORITY_CONSTANT + abs(targ - pred)
      
        self.replay_buffer.add(state, action, reward, next_state, done, priority)
        
        self.t = (self.t + 1) % UPDATE_EVERY
        
        # ensures there are enough memories to take a batch of memories
        if len(self.replay_buffer) > BATCH_SIZE:
            if self.t == 0:
                # get random batch of memories from the replay buffer
                sampled_experiences = self.replay_buffer.sample()
                # learn from the batch of memories
                self.learn(sampled_experiences)

# NEED TO KNOW WHAT DATA STRUCTURE TO USE FOR MEMORY 
# TO DO: MAKE PRIORITIZED REPLAY OPTION
class PrioritizedReplayBuffer(ReplayBuffer):
    
    def __init__(self, buffer_size, batch_size, seed):
        super().__init__(buffer_size, batch_size, seed)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])

    
    
    def getWeightsFromMemory(self, priorities):
        
        denominator = float(np.sum(np.array(priorities)**PRIORITY_A))
        wts = []
        for priority in priorities:
            numerator = float(priority**PRIORITY_A)
            wts.append(numerator/denominator)
        return wts   
    
    # returns minibatch of experiences from memory
    def sample(self):
        
        # TO DO: change to pick based on priority instead of just doing it uniformly randomly 
#         experiences = random.sample(self.memory, k=self.batch_size)
        
        wts = self.getWeightsFromMemory([e.priority for e in self.memory if e is not None])
        experiences = random.choices(self.memory, weights=wts, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done*1 for e in experiences if e is not None])).float().to(device)
#         priorities = torch.from_numpy(np.vstack([e.priority for e in experiences if e is not None])).float().to(device)
        
        
        return (states, actions, rewards, next_states, dones)
    
    # adds an experience tuple to the memory data structure
    def add(self, state, action, reward, next_state, done, priority):
        e = self.experience(state, action, reward, next_state, done, priority)
        
#         # replace old tuple with new experience with updated priority
#         sars = [x[:-1] for x in self.memory]
#         pos = 0
#         for x in sars:
#             if (list(x[0]) == list(e[0])) and (x[1] == e[1]) and (x[2] == e[2]) and (list(x[3]) == list(e[3])) and (x[4] == e[4]):
#                 del self.memory[pos]
#                 break
#             pos += 1
        
        self.memory.append(e)
        
        
        
        
        
        
        
        
        
        