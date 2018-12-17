
# Implementations of Deep Reinforcement Learning Algorithms

This repository contains PyTorch implementations of deep reinforcement learning algorithms. 


### **Algorithms Implemented (so far)** 

1. Deep Q Learning
1. Deep Q Learning with Fixed Q Targets
1. Double Deep Q Learning
1. Double Deep Q Learning with Prioritised Experience Replay
1. REINFORCE
1. Hill Climbing
7. Genetic Evolution

I plan to include PPO, DDPG and A2C soon.

### Usage ###

The algorithms are found in the Agent folder. To use the algorithms with a particular game you first create an Environment class 
to represent your game. The environment class you create should extend the Base_Environment class found in the Environments folder.  

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

### **Results**

Because results can vary greatly each run, each agent plays the game 10 times and we show the *median* result. 
We show the results in terms of number of episodes taken to reach the required score
and also time taken. The algorithms were run on a 2017 Macbook Pro (no GPUs were used).

#### **1. Cart Pole**



Below shows the number of episodes taken and also time taken for each algorithm to achieve the solution score for the game Cart Pole. The hyperparameters used are shown in the file Results/Cart_Pole/Results.py.   
 
 ![Cart Pole Results](Results/Cart_Pole/Results_Graph.png)
  

