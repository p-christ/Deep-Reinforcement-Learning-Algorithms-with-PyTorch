import torch
import torch.nn as nn
import torch.nn.functional as F


# class NN(nn.Module):
#
class NN:
    
    def __init__(self, state_size, action_size, seed, hyperparameters):
        # nn.Module.__init__(self)
        # self.seed = torch.manual_seed(seed)
        # self.fc1 = nn.Linear(state_size, hyperparameters['fc_units'][0])
        # self.fc2 = nn.Linear(hyperparameters['fc_units'][0], hyperparameters['fc_units'][1])
        # self.fc3 = nn.Linear(hyperparameters['fc_units'][1], action_size)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        self.state_size = state_size
        self.action_size = action_size
        self.hyperparameters = hyperparameters

        self.model_layers = self.create_model_layers()

        # details = [torch.nn.Linear(state_size, hyperparameters['fc_units'][0]),
        #                     torch.nn.ReLU(),
        #                     torch.nn.Linear(hyperparameters['fc_units'][0], hyperparameters['fc_units'][1]),
        #                     torch.nn.ReLU(),
        #                     torch.nn.Linear(hyperparameters['fc_units'][1], action_size)]

        self.model = torch.nn.Sequential(*self.model_layers).to(self.device)

    def get_model(self):
        return self.model


    def create_model_layers(self):

        model_layers = []

        input_dim = self.state_size

        for layer_num in range(self.hyperparameters["nn_layers"] - 1):

            output_dim = int(self.hyperparameters['nn_start_units'] * self.hyperparameters['nn_unit_decay'] ** layer_num)

            layer = torch.nn.Linear(input_dim, output_dim)
            model_layers.append(layer)
            model_layers.append(torch.nn.ReLU())

            input_dim = output_dim

        output_dim = self.action_size
        layer = torch.nn.Linear(input_dim, output_dim)

        model_layers.append(layer)

        return model_layers




    def forward(self, state):

        output = self.model(state)
        return output

        #
        #
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # output = self.fc3(x)
        #
        # return output