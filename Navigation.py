#!/usr/bin/env python
# coding: utf-8

# # Navigation
# 
# ---
# 
# You are welcome to use this coding environment to train your agent for the project.  Follow the instructions below to get started!
# 
# ### 1. Start the Environment
# 
# Run the next code cell to install a few packages.  This line will take a few minutes to run!

# In[ ]:





# In[2]:


get_ipython().system('pip -q install ./python')


# In[2]:


get_ipython().system('pip install --upgrade tornado==5.1.1')


# The environment is already saved in the Workspace and can be accessed at the file path provided below.  Please run the next code cell without making any changes.

# In[3]:


from unityagents import UnityEnvironment
import numpy as np

# please do not modify the line below
env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[4]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### 2. Examine the State and Action Spaces
# 
# Run the code cell below to print some information about the environment.

# In[4]:


# # reset the environment
# env_info = env.reset(train_mode=True)[brain_name]

# # number of agents in the environment
# print('Number of agents:', len(env_info.agents))

# # number of actions
# action_size = brain.vector_action_space_size
# print('Number of actions:', action_size)

# # examine the state space 
# state = env_info.vector_observations[0]
# print('States look like:', state)
# state_size = len(state)
# print('States have length:', state_size)


# ### 3. Take Random Actions in the Environment
# 
# In the next code cell, you will learn how to use the Python API to control the agent and receive feedback from the environment.
# 
# Note that **in this coding environment, you will not be able to watch the agent while it is training**, and you should set `train_mode=True` to restart the environment.

# In[5]:


# env_info = env.reset(train_mode=True)[brain_name] # reset the environment
# state = env_info.vector_observations[0]            # get the current state
# score = 0                                          # initialize the score
# while True:
#     action = np.random.randint(action_size)        # select an action
#     env_info = env.step(action)[brain_name]        # send the action to the environment
#     next_state = env_info.vector_observations[0]   # get the next state
#     reward = env_info.rewards[0]                   # get the reward
#     done = env_info.local_done[0]                  # see if episode has finished
#     score += reward                                # update the score
#     state = next_state                             # roll over the state to next time step
#     if done:                                       # exit loop if episode finished
#         break
    
# print("Score: {}".format(score))


# When finished, you can close the environment.

# In[6]:


# env.close()


# ### 4. It's Your Turn!
# 
# Now it's your turn to train your own agent to solve the environment!  A few **important notes**:
# - When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```
# - To structure your work, you're welcome to work directly in this Jupyter notebook, or you might like to start over with a new file!  You can see the list of files in the workspace by clicking on **_Jupyter_** in the top left corner of the notebook.
# - In this coding environment, you will not be able to watch the agent while it is training.  However, **_after training the agent_**, you can download the saved model weights to watch the agent on your own machine! 

# In[ ]:


# GOALS: 
# (1) get agent working with dqn from previous assignment (DONE)
# (2) get agent working after attempting to implement double dqn
# (3) get agent working after attempting prioritized replay on top of double dqn




# In[5]:


# INITIALIZE ENVIRONMENT
# env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


# In[6]:


# INITIALIZE AGENT
from agent import Agent
import random

actionSize = brain.vector_action_space_size
stateSize = len(env_info.vector_observations[0])
agent = Agent(state_size=stateSize, action_size=actionSize, seed=0)


# In[7]:


from collections import deque
import torch


# In[7]:


# DEEP Q-NETWORK

def dqn(num_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    
    
    epsilon = eps_start
    # scores to graph
    scores = []
    scores_window = deque(maxlen=100)
    
    
    for i in range(0, num_episodes):
        score = 0
        t = 0
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        while t < max_t:
            action = agent.act(state, epsilon)                      # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            t += 1
            if done:                                       # exit loop if episode finished
                break           
        scores.append(score)
        scores_window.append(score)
        
        epsilon = max(eps_end, eps_decay*epsilon)
        if i % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i-100, np.mean(scores_window)))
            torch.save(agent.predicted_Qnetwork.state_dict(), 'checkpoint_dqn.pth')
            return scores
            break
        
scores_dqn = dqn()        
        
    


# In[18]:


print(scores)


# In[10]:


import matplotlib
from matplotlib.pyplot import plot
# dqn scores
plot(range(len(scores_dqn)), scores_dqn)


# In[ ]:





# In[8]:


# INITIALIZE DDQN AGENT
from agent_ddqn import AgentDDQN

# reset environment
env_info = env.reset(train_mode=True)[brain_name]

# initialize ddqn agent
actionSize = brain.vector_action_space_size
stateSize = len(env_info.vector_observations[0])
agent_ddqn = AgentDDQN(state_size=stateSize, action_size=actionSize, seed=0)


# In[13]:


# DOUBLE DEEP Q-NETWORK
def ddqn(num_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    
    
    epsilon = eps_start
    # scores to graph
    scores = []
    scores_window = deque(maxlen=100)
    
    
    for i in range(0, num_episodes):
        score = 0
        t = 0
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        while t < max_t:
            action = agent_ddqn.act(state, epsilon)                      # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent_ddqn.step(state, action, reward, next_state, done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            t += 1
            if done:                                       # exit loop if episode finished
                break           
        scores.append(score)
        scores_window.append(score)
        
        epsilon = max(eps_end, eps_decay*epsilon)
        if i % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i-100, np.mean(scores_window)))
            torch.save(agent_ddqn.predicted_Qnetwork.state_dict(), 'checkpoint_ddqn.pth')
            return scores
            break
        
scores_ddqn = ddqn()        
        
    


# In[14]:


# ddqn scores
plot(range(len(scores_ddqn)),scores_ddqn)


# In[9]:


# INITIALIZE DDQN AGENT
from agent_ddqn_pr import AgentDDQNPR

# reset environment
env_info = env.reset(train_mode=True)[brain_name]

# initialize ddqn agent
actionSize = brain.vector_action_space_size
stateSize = len(env_info.vector_observations[0])
agent_ddqn_pr = AgentDDQNPR(state_size=stateSize, action_size=actionSize, seed=0)


# In[ ]:


# DOUBLE DEEP Q-NETWORK WITH PRIORITIZED REPLAY
def ddqn_pr(num_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    
    
    epsilon = eps_start
    # scores to graph
    scores = []
    scores_window = deque(maxlen=100)
    
    
    for i in range(0, num_episodes):
        score = 0
        t = 0
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        while t < max_t:
            action = agent_ddqn_pr.act(state, epsilon)                      # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent_ddqn_pr.step(state, action, reward, next_state, done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            t += 1
            if done:                                       # exit loop if episode finished
                break           
        scores.append(score)
        scores_window.append(score)
        
        epsilon = max(eps_end, eps_decay*epsilon)
        if i % 5 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i-100, np.mean(scores_window)))
            torch.save(agent_ddqn_pr.predicted_Qnetwork.state_dict(), 'checkpoint_ddqn_pr.pth')
            return scores
            break
        
scores_ddqn_pr = ddqn_pr()        
        
    


# In[ ]:





# In[ ]:





# In[ ]:




