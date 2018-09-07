import torch
import torch.nn as nn
import torch.nn.functional as F


"""WIP implementation of a duelling q network. Not finished yet"""

class Duelling_Q_Network(nn.Module):

    def __init__(self, state_size, action_size, seed, hyperparameters):
        nn.Module.__init__(self)
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hyperparameters['fc_units'][0])
        self.fc2 = nn.Linear(hyperparameters['fc_units'][0], hyperparameters['fc_units'][1])
        self.fc3_advantages = nn.Linear(hyperparameters['fc_units'][1], action_size)
        self.fc3_state_value = nn.Linear(hyperparameters['fc_units'][1], 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        advantages = self.fc3_advantages(x)
        state_values = self.fc3_state_value(x)

        output = advantages.add(state_values)

        return output