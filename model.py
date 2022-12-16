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
from utils import *



class common_net(nn.Module):
    ''' This class contains the common NN structure to the actor, critic and feature extraction networks. '''
    def __init__(self):
        super(common_net, self).__init__()
        self.conv1 = nn.Conv2d(n_frames, cnn['mid_feat'], kernel_size = cnn['kernel_size1'], stride = cnn['stride1'])
        self.conv2 = nn.Conv2d(cnn['mid_feat'], cnn['last_feat'], kernel_size = cnn['kernel_size2'], stride = cnn['stride2'])
        self.conv3 = nn.Conv2d(cnn['last_feat'], cnn['last_feat'], kernel_size = cnn['kernel_size3'], stride = cnn['stride3'])
        self.flat = Flatten()


    def forward(self, x):
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))

        state = self.flat(state)

        return state


class actor_net(nn.Module):
    ''' Actor network to estimate the action probability distribution '''
    def __init__(self, n_actions):
        super(actor_net, self).__init__()
        
        self.common = common_net()

        self.linear1 = nn.Linear(cnn['linear_input'], cnn['linear_hidden_size']) # 32 is used as an intermidiate layer
        self.linear2 = nn.Linear(cnn['linear_hidden_size'], n_actions) 
        
        
    
    def forward(self, state):
        
        state = self.common(state)
        state = F.relu(self.linear1(state))
        state = self.linear2(state)

        log_prob = F.softmax(state, dim = -1)

        return log_prob
    

class critic_net(nn.Module):
    ''' Critic network to estimate the value function of a state'''
    def __init__(self):
        super(critic_net, self).__init__()
        self.common = common_net()

        self.linear1 = nn.Linear(cnn['linear_input'], cnn['linear_hidden_size']) # 32 is used as an intermidiate layer
        self.linear2 = nn.Linear(cnn['linear_hidden_size'], 1) 
    
    
    def forward(self, state):
        
        state = self.common(state)

        state = F.relu(self.linear1(state))
        value = self.linear2(state)

        return value


# This architecture is the same as used in https://github.com/jcwleo/curiosity-driven-exploration-pytorch/blob/master/model.py, but without the LeakyRelu.
class state_encoding_net(nn.Module):
    ''' State encoding network, to estimate the state encoding, given the state representation '''
    def __init__(self):
        super(state_encoding_net, self).__init__()
        self.common = common_net()

        linear_feat = state_net['lin_feat']

        self.linear = nn.Linear(linear_feat, state_net['out_size'])  # States has 512 dimenion.

    def forward(self, state):

        state = self.common(state)

        state = F.leaky_relu(self.linear(state))

        return state 

class inverse_model(nn.Module):
    ''' This class is the network core of the self-supervised technique. This tries to predict what is the action that transitioned a state s to s'. '''
    def __init__(self):
        super(inverse_model, self).__init__()
        self.linear1 = nn.Linear(state_net['lin_feat']*2, state_net['lin_feat']) # This layer takes as input the concatenation of two states.
        self.linear2 = nn.Linear(state_net['lin_feat'], n_actions)

    def forward(self, states):

        encode = F.relu(self.linear1(states))
        prediction =  self.linear2(encode)

        action = F.softmax(prediction, dim = -1)
        
        return action

class forward_model(nn.Module):
    ''' This class defines the network that tries to predict the next state representation, given the action, and the current state representation. '''
    def __init__(self):
        super(forward_model, self).__init__()
        self.emb = nn.Embedding(n_actions, embed_dim)

        self.linear = nn.Linear(embed_dim + state_net['out_size'], state_net['out_size'])

    def forward(self, state, action):

        embedded_action = self.emb(action)
        input = torch.cat((state, embedded_action), dim = 1)

        output = F.leaky_relu(input)

        return output        





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
        self.n_actors = n_actors

        # NETWORK INIT. #
        self.actor = actor_net(n_actions = n_actions) 
        self.critic = critic_net()
        self.model_name = model_name
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
            
            args_outer = Rollout_arguments()
            print(i_episode)
            # init episode
            state = env.reset()
            

            

            new_state = state[0] #Versioning problems
            new_done = done = False
            # ROLLOUT STEPS #
            args_inner = Rollout_arguments()
            for actor in range(self.n_actors):
                
                for m in range(self.M):

                    state = new_state
                    done = new_done
                    action_sampled, action_logprob, value, states = self.rollout_act(state)

                    new_state, reward, new_done, _, _ = env.step(action_sampled.item())

                    args_inner.rewards = torch.cat((args_inner.rewards,torch.tensor([reward]).clone().detach()))
                    args_inner.states = torch.cat((args_inner.states,states.clone().detach()))
                    args_inner.actions = torch.cat((args_inner.actions,action_sampled.clone().detach()))

                    args_inner.values = torch.cat((args_inner.values,value.view(-1).clone().detach()))
                    args_inner.dones = torch.cat((args_inner.dones, torch.tensor([done]).clone().detach()))

                    args_inner.logP = torch.cat((args_inner.logP,action_logprob.clone().detach()))


                bootstrap = self.forward(self.stack_frames(new_state), who = 'critic')
                args_inner.dones = torch.cat((args_inner.dones,torch.tensor([new_done])))
            
                args_inner.advantages = self.compute_gae(args_inner.rewards.tolist(), args_inner.values.tolist(), bootstrap, args_inner.dones.tolist(), self.gamma, self.lam) # Compute advantages using the GAE
                
                args_inner.R = [args_inner.advantages[i].item() + args_inner.values[i].item() for i in range(len(args_inner.advantages))] # R is also importnat in the formulas.
                    
                args_outer.rewards = torch.cat((args_outer.rewards, args_inner.rewards.unsqueeze(0)))
                args_outer.states = torch.cat((args_outer.states, args_inner.states.unsqueeze(0)))
                args_outer.actions = torch.cat((args_outer.actions, args_inner.actions.unsqueeze(0)))
                args_outer.values = torch.cat((args_outer.values, args_inner.values.unsqueeze(0)))
                args_outer.dones = torch.cat((args_outer.dones, args_inner.dones.unsqueeze(0)))
                args_outer.logP = torch.cat((args_outer.logP, args_inner.logP.unsqueeze(0)))
                args_outer.R = torch.cat((args_outer.R, torch.tensor(args_inner.R).unsqueeze(0)))
                args_outer.advantages = torch.cat((args_outer.advantages, torch.tensor(args_inner.advantages).unsqueeze(0)))

                args_inner = Rollout_arguments()

                state = env.reset()
                new_state = state[0] #Versioning problems
                self.states = deque(maxlen = self.n_frames) # Re-initialize
            
            for n_actor in range(args_outer.advantages.shape[0]):

                args = Rollout_arguments()
                args.states = args_outer.states[n_actor].clone()
                args.actions = args_outer.actions[n_actor].clone()
                args.values = args_outer.values[n_actor].clone()
                args.R = args_outer.R[n_actor].clone()
                args.logP = args_outer.logP[n_actor].clone()
                args.advantages = args_outer.advantages[n_actor].clone()
                # Initialize the Buffer   
                buffer = BufferDataset(args)
                buffer_dataset = DataLoader(buffer, self.batch_size, shuffle = False)
                
                self.train()
                for epoch in range(self.n_epochs):
                   
                    for batch in buffer_dataset:

                        advantages_new = self.normalize_advantages(batch[Buffer['advantages']].to(self.device))  
                        logprob_new, value_new, entropy = self.optimization_act(batch[Buffer['states']].to(self.device), batch[Buffer['actions']].to(self.device)) 

                        
                        ratio = torch.exp(logprob_new - batch[Buffer['LogProb']])   #index 3 stands for the logprob
                       

                    
                        p1 = ratio * advantages_new
                        p2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantages_new
                        Lpi = torch.min(p1, p2) # Clipped loss

                        #MSE loss
                        Lv = self.MseLoss(value_new.view(-1), batch[Buffer['Return']].to(self.device)) if torch.cuda.is_available() else self.MseLoss(value_new.view(-1), batch[Buffer['Return']].to(self.device))
                    
                        #Entropy loss
                        Ls = entropy
                        # final loss of clipped objective PPO
                        loss = -Lpi - self.c2*Ls + self.c1*Lv 
                        # take gradient step
                        self.optimizer.zero_grad()
                        loss.mean().backward()
                        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                        self.optimizer.step()
                        self.scheduler.step()
              
            
            self.eval()
            if i_episode%1 == 0:
                
                
                mean_reward = evaluate_agent(self, n_eval_episodes = 1)
                if mean_reward > self.maximum:
                    self.maximum = mean_reward
                    self.save(self.model_name)
                print("The reward at episode {} is {:.4f}".format(i_episode, mean_reward))
            
            
            # evaluation step
        print('The best model has achieved {} as reward....'.format(self.maximum))
        

    def save(self, model_name = model_name):
        torch.save(self.state_dict(), model_name)

    def load(self, model_name = model_name):
        self.load_state_dict(torch.load(model_name, map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret




class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)




        


