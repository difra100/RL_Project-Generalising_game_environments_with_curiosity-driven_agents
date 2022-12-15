from torchvision import transforms
import gym
from variables import *
from PIL import Image
import torch


env = gym.make(game)
transform = transforms.Compose([  # Transformation are just the gray scale and a resizing to 64,64 
                transforms.Grayscale(),
                transforms.Resize((64,64))
            ])
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