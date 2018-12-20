
# Implementations of Deep Reinforcement Learning Algorithms

This repository contains PyTorch implementations of deep reinforcement learning algorithms. 


### **Algorithms Implemented (so far)** 

1. Deep Q Learning
1. Deep Q Learning with Fixed Q Targets
1. Double Deep Q Learning
1. Double Deep Q Learning with Prioritised Experience Replay
1. REINFORCE
1. DDPG
1. Hill Climbing
7. Genetic Evolution

All implementations are able to quickly solve either Cart Pole (discrete actions) or Mountain Car Continuous (continuous actions) and Unity's Tennis game. I plan to add PPO and A2C soon.


### **Algorithm Performance**

#### a) Cart Pole (Discrete Action Game)

Below shows the number of episodes taken and also time taken for each algorithm to achieve the solution score for the game Cart Pole. Because results can vary greatly each run, each agent plays the game 10 times and we show the *median* result. 
We show the results in terms of number of episodes taken to reach the required score
and also time taken. The algorithms were run on a 2017 Macbook Pro (no GPUs were used). The hyperparameters used are shown in the file `Results/Cart_Pole/Results.py`.   
 
![Cart Pole Results](Results/Cart_Pole/Results_Graph.png)


#### b) Mountain Car (Continuous Action Game)
  
Here are the results for DDPG with respect to the Mountain Car (Continuous) game. The algorithms were run on a 2017 Macbook Pro (no GPUs were used). The hyperparameters used are shown in the file `Results/Mountain_Car_Continuous/Results.py`.

![Mountain Car Continuous Results](Results/Mountain_Car_Continuous/Results_Graph.png)

#### c) Tennis (Continuous Action Game)

Below shows the results of a DDPG agent solving a [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment, where two agents control rackets to bounce a ball over a net. 
<p align="center"><img src="https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif" alt="Example" width="50%" style="middle"></p>

DDPG results (using a GPU):


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

And then to watch them learn Cart Pole run:
`python Results/Cart_Pole/Results.py`

To watch them learn Mountain Car run: `python Results/Mountain_Car_Continuous/Results.py`

To watch them learn Tennis run: `python Results/Tennis/Results.py`

#### ii) To Train the Agents on your Own Game  

To use the algorithms with your own particular game instead you follow these steps:
 
1. Create an Environment class to represent your game - the environment class you create should extend the `Base_Environment` class found in the `Environments` folder to make 
it compatible with all the agents.  

2. Create a config object with the hyperparameters and game you want to use. See `Results/Cart_Pole/Results.py` for an example of this.
3. Use function `run_games_for_agents` to have the different agents play the game. Again see `Results/Cart_Pole/Results.py` for an example of this.
