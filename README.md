# RL_Project:  Generalizing Game Environments through Curiosity-Driven Exploration
## Project abstract

This project aims to develop a curiosity-driven agent that learns a policy for a specific game, which can be later extended to similar, but different ones.  
This work assesses the validity and efficacy of the methodologies introduced in the [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/pdf/1705.05363v1.pdf) paper. The test games haven't been tested in the original research, and concern ATARI games such as Space invaders or other similar shoot ’em up arcade games such as Phoenix and/or Assault. This is done to show and gauge if transfer learning and the feature encoding have been effective. Among the purposes of this paper there is also the comparison of extrinsic reward vs extrinsic + intrinsic rewards training.

## Project Results

The policy gradient method that was employed is the [PPO](https://arxiv.org/abs/1707.06347) (Proximal Policy Optimization) algorithm. The setting of the algorithm are specified in 'src/variables.py'.  
The most of the experiments were conducted in [Phoenix](https://en.wikipedia.org/wiki/Phoenix_(video_game)), and then with zero-shot/few-shot learning the agent were evaluated on different games, such as [Space Invaders](https://en.wikipedia.org/wiki/Space_Invaders) or [Assault](https://en.wikipedia.org/wiki/Assault_(1988_video_game)).  

![image](http://drive.google.com/uc?export=view&id=1UZFn65-qTL_EHCfiUGLf19OPNm1ybtWd)  
A detailed explaination of the results is reported in the presentation.pptx file. The agent learnt to play at Phoenix, and it shown some ability in transferring its knowledge to SpaceInvaders. This was possible to assess, by comparing a trained-from-scratch agent in SpaceInvaders against a SpaceInvaders agent pre-trained to play on Phoenix.

## How to use this repository
* To train and evaluate the agents open the 'main.ipynb' file, where it is possible to install the requested libraries.  
* The environments are from the [openAI gym ATARI](https://www.gymlibrary.dev/environments/atari/index.html) library.  
* Before training, change the variables in 'src/variables.py' as you prefer.
## Repository Organization  
* images/ : .npy files of the shoot 'em up video-games frames, used in the experiments.  
    * assault.npy  
    * phoenix.npy  
    * space_invaders.npy  
* models/ : This folder contain the pretrained models, notice that only models for Phoenix and Space Invaders are available. Further information in the 'main.ipynb' file.  
    * old_models/ : These are some additonal models, there are not any specific features. Can be tested anyway in the main.  
    * ${Name-of-the-game}\_${ifExtrinsic-Reward-Enabled}\_${ifIntrinsic-Reward-Enabled}.pt  
* src/    : Source code.  
    * model.py : PPO algorithm, and training code is contained in this script.  
    * utils.py : This script contain various utilities, such as the image preprocessing functions, and the agent evaluation functionality.  
    * variables.py : This script contains the variables to set to modify the learning strategies, but also the models' parameters, as the video-game itself.  
* main.ipynb : main of the project, here is possible to start the trainings, by loading pretrained models, or from scratch. It is also possible to evaluate the agents.  
* presentation.pptx : Presentation of the project.  

