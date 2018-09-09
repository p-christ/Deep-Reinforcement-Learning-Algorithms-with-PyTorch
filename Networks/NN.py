import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    
    def __init__(self, state_size, action_size, seed, hyperparameters):
        nn.Module.__init__(self)
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hyperparameters['fc_units'][0])
        self.fc2 = nn.Linear(hyperparameters['fc_units'][0], hyperparameters['fc_units'][1])
        self.fc3 = nn.Linear(hyperparameters['fc_units'][1], action_size)

        self.test_simpler_model = nn.Linear(state_size, 10)
        self.te1 = nn.Linear(10, action_size)

    
    def forward(self, state):
        
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # output = self.fc3(x)
        #

        x = F.relu(self.test_simpler_model(state))
        output = self.te1(x)

        return output