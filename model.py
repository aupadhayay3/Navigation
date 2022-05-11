import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# EVENTUALLY CHANGE TO ADD OTHER COMPONENTS BESIDES FULLY CONNECTED LAYERS
# FOR EXAMPLE, DROPOUT, ETC.
class DeepQNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed):
        super(DeepQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        
        
        self.fc1 = nn.Linear(self.state_size, 64)
        self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(32, 8)
        self.fc5 = nn.Linear(8, self.action_size)
        
    
    
    
    
    def forward(self, state):
        
        # make sure state tensor is batch first
        x = state.view(-1, self.state_size)
        
        # pass the state through the network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        # produce a mini-batch of action values for each action
        action_values = self.fc5(x)
        
        
        
        return action_values