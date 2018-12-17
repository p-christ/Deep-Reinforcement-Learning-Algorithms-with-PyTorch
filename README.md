
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

All implementations are able to solve either Cart Pole (discrete actions) or Mountain Car Continuous (continuous actions) in a 
reasonable amount of time. I plan to add PPO and A2C soon.

### Usage ###

The algorithms are found in the Agent folder. 

To watch all the different agents learn cartpole follow these steps:

```commandline
git clone https://github.com/p-christ/Deep_RL_Implementations.git
cd Deep_RL_Implementations

conda create --name myenvname
y
conda activate myenvname

pip3 install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:/Deep_RL_Implementations"
python Results/Cart_Pole/Results.py
``` 

To use the algorithms with your own particular game instead you follow these steps:
 
1. Create an Environment class to represent your game - the environment class you create should extend the `Base_Environment` class found in the `Environments` folder to make 
it compatible with all the agents.  

2. Create a config object with the hyperparameters and game you want to use. See `Results/Cart_Pole/Results.py` for an example of this.
3. Use function `run_games_for_agents` to have the different agents play the game. Again see `Results/Cart_Pole/Results.py` for an example of this.

### **Algorithm Performance**

Because results can vary greatly each run, each agent plays the game 10 times and we show the *median* result. 
We show the results in terms of number of episodes taken to reach the required score
and also time taken. The algorithms were run on a 2017 Macbook Pro (no GPUs were used).

Below shows the number of episodes taken and also time taken for each algorithm to achieve the solution score for the game Cart Pole. The hyperparameters used are shown in the file `Results/Cart_Pole/Results.py`.   
 
![Cart Pole Results](Results/Cart_Pole/Results_Graph.png)
  
Here are the results for DDPG with respect to the Mountain Car (Continuous) game. The hyperparameters used are shown in the file `Results/Mountain_Car_Continuous/Results.py`.

![Mountain Car Continuous Results](Results/Mountain_Car_Continuous/My_Results_Graph.png)

