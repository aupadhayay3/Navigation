# Imports
import numpy as np
import random
from collections import namedtuple, deque

from model import DeepQNetwork 
from agent import Agent

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


class AgentDDQN(Agent):
    
    def __init__(self, state_size, action_size, seed):
        super().__init__(state_size, action_size, seed)
        
        
    # update the deep Q network based on experiences
    def learn(self, experiences, gamma=GAMMA_VALUE):
        
        # unpack experiences
        states, actions, rewards, next_states, dones = experiences
        
        # use states and actions to produce the q value from the current network
        action_values = self.predicted_Qnetwork.forward(states)
        predictedQ = action_values.gather(1, actions).squeeze()
        
        # use rewards and next_states to produce the q value from the target network
        a = self.target_Qnetwork.forward(next_states).detach()
        
        # get action from predicting network
        # instead of grabbing the q-value, grab the index of the max, indicating the action
        target_input_actions = self.predicted_Qnetwork.forward(next_states).detach().max(1)[1].unsqueeze(1)
        
        # evalutate using target network
        target_action_values = self.target_Qnetwork.forward(next_states).detach().gather(1,target_input_actions).squeeze()
        
#         target_action_values = self.target_Qnetwork.forward(next_states).detach().max(1)[0]

        # get resulting target
        targetQ = rewards.squeeze() + gamma*target_action_values.squeeze()*((1-dones).squeeze())
        
        
        # use gradient descent to update the network 
        self.optimizer.zero_grad()
        loss = F.mse_loss(predictedQ, targetQ)
        loss.backward()
        self.optimizer.step()
        
        
        # soft update target network  
        self.update_target(self.predicted_Qnetwork, self.target_Qnetwork, TAU)    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        