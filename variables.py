# Variables are freely inspired by https://github.com/DarylRodrigo/rl_lib/tree/master/PPO, and the original PPO paper https://arxiv.org/pdf/1707.06347.pdf

game = "ALE/Phoenix-v5"

model_name = game[4:] + '.pt'
seed = 30
gamma = 0.99
lamb = 0.95
n_frames = 6
n_epochs = 20
batch_size = 32
loss_eps = 0.1 #eps for the clip of the loss # ATARI Games uses 0.1
M = 128 # rollout steps
c1 = 1 # Lv weight in the objective
c2 = 0.01 # Entropy weight in the objective
training_episodes = 1000


lr = 0.00025 # Adam optimizer learning rate

# Actor and critic are mostly the same, here are the cnn hyperparameters.

# details in https://medium.com/nerd-for-tech/reinforcement-learning-deep-q-learning-with-atari-games-63f5242440b1

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


Buffer ={'states' : 0,
         'Return' : 1,
         'actions' : 2,
         'LogProb' : 3,
         'values' : 4,
         'advantages' : 5}

# This dictionary contains the values to remove the irrelevant/non-informative features from the images/state representation. 
transform_diz = {
                        'ALE/Phoenix-v5': (17, 179),   
                        'ALE/SpaceInvaders-v5': (20,195),
                        'ALE/Assault-v5': (10, 187)
}



