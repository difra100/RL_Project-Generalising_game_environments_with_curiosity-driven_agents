# Variables are freely inspired by https://github.com/DarylRodrigo/rl_lib/tree/master/PPO, and the original PPO paper https://arxiv.org/pdf/1707.06347.pdf

''' SETTING VARIABLES '''
wb = True
game = "ALE/Phoenix-v5"
project_name = 'RL_Curiosity_agent'

# This dictionary contains the values to remove the irrelevant/non-informative features from the images/state representation. 
transform_diz = {
                        'ALE/Phoenix-v5': (17, 179),   
                        'ALE/SpaceInvaders-v5': (20,195),
                        'ALE/Assault-v5': (10, 187)
                }

''' TRAINING VARIABLES '''
# model_name = 'only_intrinsic_agent.pt'#game[4:] + '.pt'
# seed = 30
gamma = 0.99
lamb = 0.95
n_frames = 6
n_epochs = 10
batch_size = 64
loss_eps = 0.1 #eps for the clip of the loss # ATARI Games uses 0.1
M = 256 # rollout steps
patience = 128
c1 = 1 # Lv weight in the objective
c2 = 0.01 # Entropy weight in the objective
training_episodes = 1000
n_actors = 8
n_steps = ((M/batch_size)*n_actors)*training_episodes
lr = 0.00025 # Adam optimizer learning rate
eta = 0.01
lr_annealing = True
eps_annealing = True
w1 = 1
w2 = 0.5
# ext = False
# intr = True




# Actor and critic are mostly the same, here are the cnn hyperparameters.

# details in https://medium.com/nerd-for-tech/reinforcement-learning-deep-q-learning-with-atari-games-63f5242440b1

''' ARCHITECTURES VARIABLES '''

config = {

        'n_frames': n_frames,
        'epochs': n_epochs,
        'batch_size': batch_size,
        'rollout_steps': M,
        'episodes': training_episodes,
        'actors': 8
}






cnn = {'mid_feat': 32,
        'last_feat': 64,
        'kernel_size1': 8,
        'kernel_size2': 4,
        'kernel_size3': 3,
        'stride1': 4,
        'stride2': 2,
        'stride3': 1,
        'linear_input': 1024,
        'linear_hidden_size': 64}


state_net = {'lin_feat': 1024,
             'out_size': 512    
            }

embed_dim = 32


Buffer ={'states' : 0,
         'Return' : 1,
         'actions' : 2,
         'LogProb' : 3,
         'values' : 4,
         'advantages' : 5,
         'next_states' : 6}





