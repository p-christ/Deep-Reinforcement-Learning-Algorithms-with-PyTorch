import torch
import torch.nn as nn
import torch.nn.functional as F


"""WIP implementation of a duelling q network. Not finished yet"""

class Duelling_NN(nn.Module):

    def __init__(self, state_size, action_size, seed, hyperparameters):
        nn.Module.__init__(self)
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hyperparameters['fc_units'][0])

        self.advantage_fc1 = nn.Linear(hyperparameters['fc_units'][0], hyperparameters['fc_units'][1])
        self.state_values_fc1 = nn.Linear(hyperparameters['fc_units'][0],hyperparameters['fc_units'][1] )

        self.advantage_fc2 = nn.Linear(hyperparameters['fc_units'][1], action_size)
        self.state_values_fc2 = nn.Linear(hyperparameters['fc_units'][1], 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))

        advantages = F.relu(self.advantage_fc1(x))
        state_values = F.relu(self.state_values_fc1(x))

        advantages = self.advantage_fc2(advantages)
        state_values = self.state_values_fc2(state_values)

        output = advantages + state_values  - advantages.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        # TO DO: check this final equation

        return output