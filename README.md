# Deep Reinforcement Learning Algorithms with PyTorch

![Travis CI](https://travis-ci.org/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch.svg?branch=master)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)



![RL](utilities/RL_image.jpeg)   ![PyTorch](utilities/PyTorch-logo-2.jpg)

This repository contains PyTorch implementations of deep reinforcement learning algorithms and environments. 

(To help you remember things you learn about machine learning in general write them in [Save All](https://saveall.ai/shared/deck/140&4&3K3uXPazkg4) and try out the public deck there about Fast AI's machine learning textbook.)

## **Algorithms Implemented**  

1. *Deep Q Learning (DQN)* <sub><sup> ([Mnih et al. 2013](https://arxiv.org/pdf/1312.5602.pdf)) </sup></sub>  
1. *DQN with Fixed Q Targets* <sub><sup> ([Mnih et al. 2013](https://arxiv.org/pdf/1312.5602.pdf)) </sup></sub>
1. *Double DQN (DDQN)* <sub><sup> ([Hado van Hasselt et al. 2015](https://arxiv.org/pdf/1509.06461.pdf)) </sup></sub>
1. *DDQN with Prioritised Experience Replay* <sub><sup> ([Schaul et al. 2016](https://arxiv.org/pdf/1511.05952.pdf)) </sup></sub>
1. *Dueling DDQN* <sub><sup> ([Wang et al. 2016](http://proceedings.mlr.press/v48/wangf16.pdf)) </sup></sub>
1. *REINFORCE* <sub><sup> ([Williams et al. 1992](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)) </sup></sub>
1. *Deep Deterministic Policy Gradients (DDPG)* <sub><sup> ([Lillicrap et al. 2016](https://arxiv.org/pdf/1509.02971.pdf) ) </sup></sub>
1. *Twin Delayed Deep Deterministic Policy Gradients (TD3)* <sub><sup> ([Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477)) </sup></sub>
1. *Soft Actor-Critic (SAC)* <sub><sup> ([Haarnoja et al. 2018](https://arxiv.org/pdf/1812.05905.pdf)) </sup></sub>
1. *Soft Actor-Critic for Discrete Actions (SAC-Discrete)* <sub><sup> ([Christodoulou 2019](https://arxiv.org/abs/1910.07207)) </sup></sub> 
1. *Asynchronous Advantage Actor Critic (A3C)* <sub><sup> ([Mnih et al. 2016](https://arxiv.org/pdf/1602.01783.pdf)) </sup></sub>
1. *Syncrhonous Advantage Actor Critic (A2C)*
1. *Proximal Policy Optimisation (PPO)* <sub><sup> ([Schulman et al. 2017](https://openai-public.s3-us-west-2.amazonaws.com/blog/2017-07/ppo/ppo-arxiv.pdf)) </sup></sub>
1. *DQN with Hindsight Experience Replay (DQN-HER)* <sub><sup> ([Andrychowicz et al. 2018](https://arxiv.org/pdf/1707.01495.pdf)) </sup></sub>
1. *DDPG with Hindsight Experience Replay (DDPG-HER)* <sub><sup> ([Andrychowicz et al. 2018](https://arxiv.org/pdf/1707.01495.pdf) ) </sup></sub>
1. *Hierarchical-DQN (h-DQN)* <sub><sup> ([Kulkarni et al. 2016](https://arxiv.org/pdf/1604.06057.pdf)) </sup></sub>
1. *Stochastic NNs for Hierarchical Reinforcement Learning (SNN-HRL)* <sub><sup> ([Florensa et al. 2017](https://arxiv.org/pdf/1704.03012.pdf)) </sup></sub>
1. *Diversity Is All You Need (DIAYN)* <sub><sup> ([Eyensbach et al. 2018](https://arxiv.org/pdf/1802.06070.pdf)) </sup></sub>

All implementations are able to quickly solve Cart Pole (discrete actions), Mountain Car Continuous (continuous actions), 
Bit Flipping (discrete actions with dynamic goals) or Fetch Reach (continuous actions with dynamic goals). I plan to add more hierarchical RL algorithms soon.

## **Environments Implemented**

1. *Bit Flipping Game* <sub><sup> (as described in [Andrychowicz et al. 2018](https://arxiv.org/pdf/1707.01495.pdf)) </sup></sub>
1. *Four Rooms Game* <sub><sup> (as described in [Sutton et al. 1998](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)) </sup></sub>
1. *Long Corridor Game* <sub><sup> (as described in [Kulkarni et al. 2016](https://arxiv.org/pdf/1604.06057.pdf)) </sup></sub>
1. *Ant-{Maze, Push, Fall}* <sub><sup> (as desribed in [Nachum et al. 2018](https://arxiv.org/pdf/1805.08296.pdf) and their accompanying [code](https://github.com/tensorflow/models/tree/master/research/efficient-hrl)) </sup></sub>

## **Results**

#### 1. Cart Pole and Mountain Car

Below shows various RL algorithms successfully learning discrete action game [Cart Pole](https://github.com/openai/gym/wiki/CartPole-v0)
 or continuous action game [Mountain Car](https://github.com/openai/gym/wiki/MountainCarContinuous-v0). The mean result from running the algorithms 
 with 3 random seeds is shown with the shaded area representing plus and minus 1 standard deviation. Hyperparameters
 used can be found in files `results/Cart_Pole.py` and `results/Mountain_Car.py`. 
 
![Cart Pole and Mountain Car Results](results/data_and_graphs/CartPole_and_MountainCar_Graph.png) 


#### 2. Hindsight Experience Replay (HER) Experiements

Below shows the performance of DQN and DDPG with and without Hindsight Experience Replay (HER) in the Bit Flipping (14 bits) 
and Fetch Reach environments described in the papers [Hindsight Experience Replay 2018](https://arxiv.org/pdf/1707.01495.pdf) 
and [Multi-Goal Reinforcement Learning 2018](https://arxiv.org/abs/1802.09464). The results replicate the results found in 
the papers and show how adding HER can allow an agent to solve problems that it otherwise would not be able to solve at all. Note that the same hyperparameters were used within each pair of agents and so the only difference 
between them was whether hindsight was used or not. 

![HER Experiment Results](results/data_and_graphs/HER_Experiments.png)

#### 3. Hierarchical Reinforcement Learning Experiments

The results on the left below show the performance of DQN and the algorithm hierarchical-DQN from [Kulkarni et al. 2016](https://arxiv.org/pdf/1604.06057.pdf)
on the Long Corridor environment also explained in [Kulkarni et al. 2016](https://arxiv.org/pdf/1604.06057.pdf). The environment
requires the agent to go to the end of a corridor before coming back in order to receive a larger reward. This delayed 
gratification and the aliasing of states makes it a somewhat impossible game for DQN to learn but if we introduce a 
meta-controller (as in h-DQN) which directs a lower-level controller how to behave we are able to make more progress. This 
aligns with the results found in the paper. 

The results on the right show the performance of DDQN and algorithm Stochastic NNs for Hierarchical Reinforcement Learning 
(SNN-HRL) from [Florensa et al. 2017](https://arxiv.org/pdf/1704.03012.pdf). DDQN is used as the comparison because
the implementation of SSN-HRL uses 2 DDQN algorithms within it. Note that the first 300 episodes of training
for SNN-HRL were used for pre-training which is why there is no reward for those episodes. 
 
![Long Corridor and Four Rooms](results/data_and_graphs/Four_Rooms_and_Long_Corridor.png)
     

### Usage ###

The repository's high-level structure is:
 
    ├── agents                    
        ├── actor_critic_agents   
        ├── DQN_agents         
        ├── policy_gradient_agents
        └── stochastic_policy_search_agents 
    ├── environments   
    ├── results             
        └── data_and_graphs        
    ├── tests
    ├── utilities             
        └── data structures            
   

#### i) To watch the agents learn the above games  

To watch all the different agents learn Cart Pole follow these steps:

```commandline
git clone https://github.com/p-christ/Deep_RL_Implementations.git
cd Deep_RL_Implementations

conda create --name myenvname
y
conda activate myenvname

pip3 install -r requirements.txt

python results/Cart_Pole.py
``` 

For other games change the last line to one of the other files in the Results folder. 

#### ii) To train the agents on another game  

Most Open AI gym environments should work. All you would need to do is change the config.environment field (look at `Results/Cart_Pole.py`  for an example of this). 

You can also play with your own custom game if you create a separate class that inherits from gym.Env. See `Environments/Four_Rooms_Environment.py`
for an example of a custom environment and then see the script `Results/Four_Rooms.py` to see how to have agents play the environment.
