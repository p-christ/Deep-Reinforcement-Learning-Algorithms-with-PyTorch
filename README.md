# TO DO

1. Implement prioritised experience replay properly
2. Evolution algorithm
3. Think about and start adding tests
4. Have hill climbing pick probabilistically instead of max


Later. Implement algorithms from Berkley Deep RL course


# Implementations of Deep Reinforcement Learning Algorithms

This (WIP) repository contains:

* PyTorch implementations of deep reinforcement learning algorithms
* Analysis of algorithm performance in different game environments





# 1) Deep Reinforcement Learning Algorithms Implemented

--------------------------------------------------
### A. Deep Q Learning
--------------------------------------------------

The Deep Q Learning algorithm was implemented including experience replay. Experience replay is when you store your experiences and then randomly sample from past experiences during training. The purpose of this is to decorrelate the data you are using to train your policy. 

--------------------------------------------------
### B. Deep Q Learning with Fixed Q-Targets
--------------------------------------------------

The Deep Q Learning algorithm above was extended to include Fixed Q-Targets. Using Fixed Q-Targets means that to calculate the Q-targets we use a different neural network than our policy network. The alternative neural network we use is effectively an older version of the policy. This generally has the effect of stabilising training.

Using Fixed Q-Targets with experience replay makes this algorithm similar to the algorithm used in [DeepMind's 2013 Atari Paper](https://arxiv.org/pdf/1312.5602v1.pdf)

--------------------------------------------------
### C. Doule Deep Q Learning
--------------------------------------------------

The Deep Q Learning algorithm was also extended to a Doulbe Deep Q Learning algorithm. This means that when calculating the Q-value resulting from the best action we use two different networks (one to select the maximum action and one to calculate the Q-value of that action) rather than one network. This helps alleviate over-estimation of the Q-values and is described in this [paper](https://arxiv.org/pdf/1509.06461.pdf). 


# 2) Environments Analysed

--------------------------------------------------
### A. Unity Banana Environment
--------------------------------------------------

To train the agents on the Unity Banana environment and see their results:

- Navigate to Deep_RL_Implementations/
- On the command line run: 
  - pip install -r requirements.txt
  - cd Results/Unity_Banana_Environment
  - jupyter notebook
- Then open the notebook Unity_Banana_Environment_Results.ipynb and run all cells  
- Results for the 3 agents are given below:


![Unity Banana Results](Results/Unity_Banana_Environment/unity_banana_results.png)


--------------------------------------------------
### B. Other Environments
--------------------------------------------------

TBD
