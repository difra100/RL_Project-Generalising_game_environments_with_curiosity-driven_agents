import gym

phoenix = "ALE/Phoenix-v5"


env_id = phoenix

env = gym.make(env_id, )


seed = 30
gamma = 0.99
lamb = 0.95
n_frames = 6
n_epochs = 20
batch_size = 256
MAX_PATIENCE = 100 # To avoid a learning phase with too many failures.
loss_eps = 0.2 #eps for the clip of the loss
M = 2000 # rollout steps
c1 = 0.5 # Lv weight in the objective
c2 = 0.01 # Entropy weight in the objective
training_episodes = 150
# https://github.com/elsheikh21/car-racing-ppo/blob/master/Report.pdf
lr = 0.0003 # Adam optimizer learning rate
ad_eps = 1e-5 #epsilon for adam.

# Actor and critic are mostly the same, here are the cnn hyperparameters.
cnn = {'mid_feat': 32,
        'last_feat': 64,
        'kernel_size': 5,
        'linear_input': 3136,
        'pool' : 3}


Buffer ={'states' : 0,
         'Return' : 1,
         'actions' : 2,
         'LogProb' : 3,
         'values' : 4,
         'advantages' : 5}
