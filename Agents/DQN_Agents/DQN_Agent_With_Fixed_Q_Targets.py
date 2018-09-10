from Networks.NN import NN
from Agents.DQN_Agents.DQN_Agent import DQN_Agent

from Utilities import override
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np


class DQN_Agent_With_Fixed_Q_Targets(DQN_Agent):
    
    def __init__(self, environment, seed, hyperparameters, rolling_score_length, 
                 average_score_required, agent_name):        
        DQN_Agent.__init__(self, environment=environment,
                            seed=seed, hyperparameters=hyperparameters, rolling_score_length=rolling_score_length,
                            average_score_required=average_score_required, agent_name=agent_name)

        self.qnetwork_target = NN(self.state_size, self.action_size, seed, hyperparameters).get_model()
            # .to(self.device )
            # to(self.device)

        
#     @override        
    def learn(self):        
        if self.time_to_learn():
            states, actions, rewards, next_states, dones = self.sample_experiences() #Sample experiences                        
            loss = self.compute_loss(states, next_states, rewards, actions, dones) #Compute the loss
            self.take_optimisation_step(loss) #Take an optimisation step            
            self.soft_update_of_target_network() #Update the target network
            
#     @override
    def compute_q_values_for_next_states(self, next_states):

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next
    
    def soft_update_of_target_network(self):
        """Updates the target network in the direction of the local network but by taking a step size 
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        
        tau = self.hyperparameters["tau"]
        
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)            
        
