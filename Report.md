# Description of Analysis on the Unity Banana Environment

Three different algorithms were tried on the Unity Banana Environment:

1. Deep Q Network
1. Deep Q Network with Fixed Q-Targets
1. Double Deep Q Network


The Double Q Network achieved the target average score of 13 (over 100 episodes) fastest after only **289 episodes**. The graph below shows the performance of the three algorithms:

![Unity Banana Results](Results/Unity_Banana_Environment/unity_banana_results.png)


## Details on the Implementation of the Double Deep Q Network

The algorithm works as follows:

1. A "local" neural network and "target" neural network are randomly initialised along with a memory buffer
1. The first state of an episode is provided
1. Using an epsilon-greedy policy and the local neural network an action is chosen and conducted
1. The next state, reward and whether the episode is done or not is observed
1. The whole experience is stored in the memory buffer
1. Once there is enough experience to learn from, a random sample of past experiences is sampled
1. The local network is trained using the sampled experiences. The loss used for the network is the difference between: a) Q-targets: This is equal to the reward observed plus discount rate gamma multiplied by the Q-value given by: the Q-values calculated using the "target" network and next state where the action chosen is inferred from the "local" network values. b) Q-expected = This is the Q-value given by the state and the "local" network
1. Repeat this process until convergence

The neural networks used both had 3 layers with (30, 30, 4) hidden units. 

Other hyperparameter values were:

* Learning rate: 0.0005
* Batch size: 64
* Buffer size: 100000
* Epsilon: 0.05
* Gamma: 0.99
* Tau: 0.001

