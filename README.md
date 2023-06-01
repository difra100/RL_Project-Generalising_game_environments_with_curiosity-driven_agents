# RL_Project:  Generalizing Game Environments through Curiosity-Driven Exploration
## Project abstract

This project aims to develop a curiosity-driven agent that learns a policy for a specific game, which can be later extended to similar, but different ones.  
This work assesses the validity and efficacy of the methodologies introduced in the [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/pdf/1705.05363v1.pdf) paper. The test games haven't been tested in the original research, and concern ATARI games such as Space invaders or other similar shoot ’em up arcade games such as Phoenix and/or Assault. This is done to show and gauge if transfer learning and the feature encoding have been effective. Among the purposes of this paper there is also the comparison of extrinsic reward vs extrinsic + intrinsic rewards training.

## Project Results

The algorithm that was used is the [PPO](https://arxiv.org/abs/1707.06347) (Proximal Policy Optimization) algorithm. The setting of the algorithm are specified in 'src/variables.py'.  
The most of the experiments were conducted in [Phoenix](https://en.wikipedia.org/wiki/Phoenix_(video_game)), and then with zero-shot/few-shot learning the agent were evaluated on different games, such as [Space Invaders](https://en.wikipedia.org/wiki/Space_Invaders).  

![image](http://drive.google.com/uc?export=view&id=1UZFn65-qTL_EHCfiUGLf19OPNm1ybtWd)
