from torchvision import transforms
import gym
from torch.utils.data import Dataset, DataLoader
from collections import deque
from src.variables import *
from PIL import Image
import torch

import numpy as np


env = gym.make(game)
transform = transforms.Compose([  # Transformation are just the gray scale and a resizing to 64,64 
                transforms.Grayscale(),
                transforms.Resize((64,64))
            ])

n_actions = env.action_space.n
def transform_state(state, game, device):
    ''' This function takes the numpy array state, and return its preprocessed version. 
            INPUT: state: is the numpy array, game: is the game that we are considering, there would be different transformation, based on them,
            OUTPUT: new_state: is the result of the set of transformation.
    '''

    state = state[transform_diz[game][0]:transform_diz[game][1], :, :].transpose(2,0,1)
    state = torch.from_numpy(state)

    state_new = transform(state)


    state_new = state_new/255.0


    return state_new.to(device)


class BufferDataset(Dataset): # Dataset class pytorch.

    def __init__(self, data):
        self.data = data

    def __len__(self):

        return self.data.R.shape[0]

    def __getitem__(self, idx):

        return self.data.states[idx], self.data.R[idx], self.data.actions[idx], self.data.logP[idx], self.data.values[idx], self.data.advantages[idx], self.data.next_states[idx]

def evaluate_agent(agent, n_eval_episodes = 1, render = False):

        ''' This function is useful to evaluate at the end of each iteration episode what are the current agent performances 
            INPUT: n_eval_episodes: Number of episodes used for the evaluation,
            OUTPUT: Reward statistics of the agent. '''

  
        if render:
            env = gym.make(game, render_mode = 'human')
        else:
            env = gym.make(game)

        agent.eval()
        with torch.no_grad():    
            rewards = []
            for episode in range(n_eval_episodes):
                total_reward = 0
                done = False
                s, _ = env.reset()
                # reward_deque = deque(maxlen = 100)
                while not done:
                    action = agent.act(s)
                    
                    s, reward, done, truncated, _ = env.step(action)
                    # print(reward)
                    # reward_deque.append(reward)
                    # if np.mean(reward_deque) == 0.0:
                    #     print('stuck....')
                    if render:
                        env.render()
                    total_reward += reward
                    if done or truncated: break
                
                rewards.append(total_reward)
                
                

            return sum(rewards)/len(rewards)


class Rollout_arguments:

    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.states = torch.tensor([], device = device)
        self.next_states = torch.tensor([], device = device)

        self.actions = torch.tensor([], device = device)
        self.logP = torch.tensor([], device = device)
        self.dones = torch.tensor([], device = device)
        self.rewards = torch.tensor([], device = device)
        self.values = torch.tensor([], device = device)
        self.advantages = torch.tensor([], device = device)
        self.R = torch.tensor([], device = device)

    def restore(self):

        del self.states 
        del self.actions
        del self.logP 
        del self.rewards 
        del self.values 
        del self.dones
        del self.advantages
        del self.R
        del self.next_states