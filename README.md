
# Implementations of Deep Reinforcement Learning Algorithms

This repository contains PyTorch implementations of deep reinforcement learning algorithms. 


### **Algorithms Implemented (so far)** 



1. Deep Q Learning ([Minh 2013](https://arxiv.org/pdf/1312.5602.pdf))  
1. Deep Q Learning with Fixed Q Targets ([Minh 2013](https://arxiv.org/pdf/1312.5602.pdf))
1. Double Deep Q Learning ([Hado van Hasselt 2015](https://arxiv.org/pdf/1509.06461.pdf))
1. Double Deep Q Learning with Prioritised Experience Replay ([Schaul 2016](https://arxiv.org/pdf/1511.05952.pdf))
1. REINFORCE ([Williams 1992](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf))
1. PPO ([Schulman 2017](https://openai-public.s3-us-west-2.amazonaws.com/blog/2017-07/ppo/ppo-arxiv.pdf))
1. DDPG ([Lillicrap 2016](https://arxiv.org/pdf/1509.02971.pdf)) 
1. Hill Climbing
7. Genetic Evolution

All implementations are able to quickly solve either Cart Pole (discrete actions) or Mountain Car Continuous (continuous actions) and Unity's Tennis game. I plan to add A2C and A3C soon.


### **Algorithm Performance**

#### a) Cart Pole (Discrete Action Game)

Below shows the number of episodes taken and also time taken for each algorithm to achieve the solution score for the game Cart Pole. Because results can vary greatly each run, each agent plays the game 10 times and we show the *median* result. 
We show the results in terms of number of episodes taken to reach the required score
and also time taken. The algorithms were run on a 2017 Macbook Pro (no GPUs were used). The hyperparameters used are shown in the file `Results/Cart_Pole/Results.py`.   
 
![Cart Pole Results](Results/Cart_Pole/Results_Graph.png)


#### b) Mountain Car (Continuous Action Game)
  
Here are the results for DDPG with respect to the Mountain Car (Continuous) game. The algorithms were run on a 2017 Macbook Pro (no GPUs were used). The hyperparameters used are shown in the file `Results/Mountain_Car_Continuous/Results.py`.

![Mountain Car Continuous Results](Results/Mountain_Car_Continuous/Results_Graph.png)

#### c) Tennis (Continuous Action Multi-Agent Game)

Below shows the results of a DDPG agent solving a [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment, where two agents control rackets to bounce a ball over a net. 
<p align="center"><img src="https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif" alt="Example" width="50%" style="middle"></p>

DDPG results using a GPU with the hyperparameters found in `Results/Tennis/Results.py` :


![Tennis Results](Results/Tennis/Results_Graph.png)


### Usage ###

The algorithms are found in the Agent folder. 

#### i) To Watch the Agents Learn the Above Games  

To watch all the different agents learn the above games follow these steps:

```commandline
git clone https://github.com/p-christ/Deep_RL_Implementations.git
cd Deep_RL_Implementations

conda create --name myenvname
y
conda activate myenvname

pip3 install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:/Deep_RL_Implementations"
``` 

And then to watch them learn **Cart Pole** run:
`python Results/Cart_Pole/Results.py`

To watch them learn **Mountain Car** run: `python Results/Mountain_Car_Continuous/Results.py`

To watch them learn **Tennis** you will need to download the environment:

1. Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
1. Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
1. Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
1. Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

and then run: `python Results/Tennis/Results.py`

#### ii) To Train the Agents on your Own Game  

To use the algorithms with your own particular game instead you follow these steps:
 
1. Create an Environment class to represent your game - the environment class you create should extend the `Base_Environment` class found in the `Environments` folder to make 
it compatible with all the agents.  

2. Create a config object with the hyperparameters and game you want to use. See `Results/Cart_Pole/Results.py` for an example of this.
3. Use function `run_games_for_agents` to have the different agents play the game. Again see `Results/Cart_Pole/Results.py` for an example of this.
