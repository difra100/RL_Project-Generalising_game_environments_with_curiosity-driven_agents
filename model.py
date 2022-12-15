import gym
import torch
import random
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
from torchvision import transforms
from collections import deque
from variables import *
from utility import *


class actor_net(nn.Module):
    ''' Actor network to estimate the action probability distribution '''
    def __init__(self, n_actions):
        super(actor_net, self).__init__()
        self.conv1 = nn.Conv2d(n_frames, cnn['mid_feat'], kernel_size = cnn['kernel_size1'], stride = cnn['stride1'])
        self.conv2 = nn.Conv2d(cnn['mid_feat'], cnn['last_feat'], kernel_size = cnn['kernel_size2'], stride = cnn['stride2'])
        self.conv3 = nn.Conv2d(cnn['last_feat'], cnn['last_feat'], kernel_size = cnn['kernel_size3'], stride = cnn['stride3'])
        self.linear1 = nn.Linear(cnn['linear_input'], cnn['linear_hidden_size']) # 32 is used as an intermidiate layer
        self.linear2 = nn.Linear(cnn['linear_hidden_size'], n_actions) 
        
        self.flat = Flatten()
    
    def forward(self, state):
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))

        state = self.flat(state)
        state = F.relu(self.linear1(state))
        state = self.linear2(state)

        log_prob = F.softmax(state, dim = -1)

        return log_prob
    

class critic_net(nn.Module):
    ''' Critic network to estimate the value function of a state'''
    def __init__(self):
        super(critic_net, self).__init__()
        self.conv1 = nn.Conv2d(n_frames, cnn['mid_feat'], kernel_size = cnn['kernel_size1'], stride = cnn['stride1'])
        self.conv2 = nn.Conv2d(cnn['mid_feat'], cnn['last_feat'], kernel_size = cnn['kernel_size2'], stride = cnn['stride2'])
        self.conv3 = nn.Conv2d(cnn['last_feat'], cnn['last_feat'], kernel_size = cnn['kernel_size3'], stride = cnn['stride3'])
        self.linear1 = nn.Linear(cnn['linear_input'], cnn['linear_hidden_size']) # 32 is used as an intermidiate layer
        self.linear2 = nn.Linear(cnn['linear_hidden_size'], 1) 
        
        self.flat = Flatten()
    
    def forward(self, state):
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))

        state = self.flat(state)
        state = F.relu(self.linear1(state))
        value = self.linear2(state)

        return value




class Policy(nn.Module):

    def __init__(self, load = False):
        super(Policy, self).__init__()

      
        # Get the state space and action space
        n_actions = env.action_space.n
        self.seed_everything(seed = seed)
        
        self.gamma = gamma
        self.lam = lamb
        self.n_frames = n_frames   # Number of frames to consider 
        self.n_epochs = n_epochs
        self.batch_size = batch_size # total number of sample is 128 at most.
        self.eps = loss_eps
        self.M = M #rollout steps, this is an arbitrary number dependant on the environment
        self.maximum = 0
        self.c1 = c1  # These are hyperparameters
        self.c2 = c2 # These are hyperparameters

        # NETWORK INIT. #
        self.actor = actor_net(n_actions = n_actions) 
        self.critic = critic_net()
        
        self.states = deque(maxlen = self.n_frames)
        self.optimizer = optim.Adam(self.parameters(), lr=lr) ## Play with THE LR, epsilon is due to implementation details
        self.scheduler = StepLR(self.optimizer, step_size=10000, gamma=0.85) # learning rate's scheduler.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.MseLoss = nn.MSELoss(self.device)
        self.to(self.device)
        if load:
            self.load()
    
    def forward(self, x, who = 'actor'):
        
        if who == 'critic':
            x = self.critic(x)
        else:
            x = self.actor(x)

        
        return x

    

    def stack_frames(self, state):

        state = transform_state(state, game, device = self.device) #apply the classical transformation embeddeed inside the transform_state method
        self.states.append(state) # If the last state has not been added yet
             
        if len(self.states) < self.n_frames: # If the set of consecutive states is not filled yet.
            while len(self.states) < self.n_frames:
                self.states.append(state)
        
        states = torch.vstack([state for state in self.states]).unsqueeze(0) # New pytorch represented state.
        
        return states
        
    def seed_everything(self, seed):
        ''' This function has the purpose of enhance the experiment reproducibility. '''
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def act(self, state):
        
        ''' This function is needed for the final evaluation of the agent, it returns the best action possible given the state. 
            INPUT: state: This is the numpy array state,
            OUTPUT: action_sample: We evaluate the best action at inference time. '''
        self.eval()
        with torch.no_grad():
            states = self.stack_frames(state)

            action = self.forward(states)

            action_sampled = action.argmax(-1)
            # print(action_sampled.item())
            return action_sampled.item()

    def rollout_act(self, state):
        ''' This function is needed for the rollout phase of the agent, it returns a uniform sampled action for a given the state. 
            INPUT: state: This is the numpy array state,
            OUTPUT: action_sample: We return the sampled log prob action, and the respective action. '''
        self.eval()
        with torch.no_grad():
            states = self.stack_frames(state)

            action = self.forward(states)
            value = self.forward(states, 'critic')

            dist = Categorical(action)
            action_sampled = dist.sample()
            action_logprob = dist.log_prob(action_sampled)

            return action_sampled, action_logprob, value, states

    def optimization_act(self, state, action):
        ''' This function is needed for the optimization phase of the agent, it returns a log prob of the all possible action, and the respective entropy. 
            INPUT: state: This is the numpy array state,
            OUTPUT: action_sample: We return the sampled log prob action, and the respective action. '''
     
        # the queue is already obtained during the rollout

        action_probs = self.forward(state)

        dist = Categorical(action_probs)

        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        values = self.forward(state, who = 'critic')

        return log_probs, values, entropy

    def compute_gae(self, rewards, values, bootstrap_values, terminals, gamma, lam):
        ''' This code is required to compute the advantages with GAE, it is freely inspired by https://github.com/elsheikh21/car-racing-ppo, the code was converted in a pytorch format.
            INPUT: (rewards, values, bootstrap_values, terminals, gamma, lam), these are measures bot pre-defined and obtaine dduring the process
            OUTPUT: Advantage tensor. '''
     
        values = values.copy()
        values.append(bootstrap_values)
        # Compute delta
        deltas = []
        for i in reversed(range(len(rewards))):

            V = rewards[i] + (1.0 - terminals[i]) * gamma * values[i + 1]
            delta = V - values[i]
            deltas.append(delta)
        deltas = torch.tensor(list(reversed(deltas)), device = self.device)
       
        # Compute gae
        A = deltas[-1]
        advantages = [A]
        for i in reversed(range(len(deltas) - 1)):
            A = deltas[i] + (1.0 - terminals[i]) * gamma * lam * A
            advantages.append(A)
        advantages = reversed(advantages)

        return torch.tensor(list(advantages), device = self.device)

    
    def normalize_advantages(self, advantages):
        eps = 1e-7
        advantages = (advantages - advantages.mean())/(advantages.std() + eps) 

        return advantages


    def trainer(self, n_training_episodes=training_episodes):
       
         # Lists of parameters to save in the meanwhile
        # Optimizer definition #

        for i_episode in range(1, n_training_episodes+1):
            args = Rollout_arguments()
            print(i_episode)
            # init episode
            state = env.reset()
            

            for i in range(60): # Noisy start, Same function from the practical
                state,_,_,_,_ = env.step(0)

            new_state = state #[0] Versioning problems
            done_pat = False
            new_done = done = False
            # ROLLOUT STEPS #
            while not done_pat:
                for m in range(self.M):

                    state = new_state
                    done = new_done
                    action_sampled, action_logprob, value, states = self.rollout_act(state)

                    new_state, reward, new_done, _, _ = env.step(action_sampled.item())

                    args.rewards.append(reward)
                    args.states.append(states.detach().squeeze(0))
                    args.actions.append(action_sampled.detach().item())
                    args.values.append(value.detach())
                    args.dones.append(done)

                    args.logP.append(action_logprob.detach())

                done_pat = True

                
            bootstrap = self.forward(self.stack_frames(new_state), who = 'critic')
            args.dones.append(new_done)
           
            args.advantages = self.compute_gae(args.rewards, args.values, bootstrap, args.dones, self.gamma, self.lam) # Compute advantages using the GAE
            args.R = [args.advantages[i].item() + args.values[i].item() for i in range(len(args.advantages))] # R is also importnat in the formulas.
        
            self.states = deque(maxlen = self.n_frames) # Re-initialize

            # Initialize the Buffer   
            buffer = BufferDataset(args)
            buffer_dataset = DataLoader(buffer, self.batch_size, shuffle = False)
            self.train()
            for epoch in range(self.n_epochs):
                for batch in buffer_dataset:
               

                    advantages_new = self.normalize_advantages(batch[Buffer['advantages']].to(self.device))  
                    logprob_new, value_new, entropy = self.optimization_act(batch[Buffer['states']].to(self.device), batch[Buffer['actions']].to(self.device)) 

                    ratio = torch.exp(logprob_new.reshape(-1,1) - batch[Buffer['LogProb']])   #index 3 stands for the logprob
                    advantages_new = advantages_new.reshape(-1,1).detach()
                    

                
                    p1 = ratio * advantages_new
                    p2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantages_new
                    Lpi = torch.min(p1, p2) # Clipped loss

                    #MSE loss
                    Lv = self.MseLoss(value_new.to(torch.float64).view(-1), batch[Buffer['Return']].to(self.device)) if torch.cuda.is_available() else self.MseLoss(value_new.to(torch.float64).view(-1), batch[Buffer['Return']].to(self.device))
                  
                    #Entropy loss
                    Ls = entropy
                    # final loss of clipped objective PPO
                    loss = -Lpi - self.c2*Ls + self.c1*Lv 
                    # take gradient step
                    self.optimizer.zero_grad()
                    loss.mean().backward()

                    self.optimizer.step()
                    self.scheduler.step()
            self.eval()
            if i_episode%1 == 0:
                
                
                mean_reward = evaluate_agent(self, n_eval_episodes = 1)
                if mean_reward > self.maximum:
                    self.maximum = mean_reward
                    self.save()
                print("The reward at episode {} is {:.4f}".format(i_episode, mean_reward))
            
            
            # evaluation step
        print('The best model has achieved {} as reward....'.format(self.maximum))
        

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret




class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class Rollout_arguments:

    def __init__(self):
        self.states = []
        self.actions = []
        self.logP = []
        self.dones = []
        self.rewards = []
        self.values = []

    def restore(self):

        del self.states 
        del self.actions
        del self.logP 
        del self.rewards 
        del self.values 
        del self.dones

class BufferDataset(Dataset): # Dataset class pytorch.

    def __init__(self, data):
        self.data = data

    def __len__(self):

        return len(self.data.rewards)

    def __getitem__(self, idx):

        return self.data.states[idx], self.data.R[idx], self.data.actions[idx], self.data.logP[idx], self.data.values[idx], self.data.advantages[idx]

def evaluate_agent(agent, n_eval_episodes = 5, render = False):

        ''' This function is useful to evaluate at the end of each iteration episode what are the current agent performances 
            INPUT: n_eval_episodes: Number of episodes used for the evaluation,
            OUTPUT: Reward statistics of the agent. '''

  
        max_steps_per_episode = 1000
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
                for i in range(max_steps_per_episode):
                    action = agent.act(s)
                    
                    s, reward, done, truncated, _ = env.step(action)
                    if render:
                        env.render()
                    total_reward += reward
                    if done or truncated: break
                
                rewards.append(total_reward)
                
                

                return sum(rewards)/len(rewards)
        


