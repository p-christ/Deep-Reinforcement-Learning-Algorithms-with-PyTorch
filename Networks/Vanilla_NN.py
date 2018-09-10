import torch
import torch.nn as nn
import torch.nn.functional as F


class Vanilla_NN:
    
    def __init__(self, state_size, action_size, seed, hyperparameters):
        self.seed = torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.hyperparameters = hyperparameters

        self.model_layers = self.create_model_layers()

        self.model = torch.nn.Sequential(*self.model_layers)

    def create_model_layers(self):

        model_layers = []

        input_dim = self.state_size

        for layer_num in range(self.hyperparameters["nn_layers"] - 1):

            output_dim = int(self.hyperparameters['nn_start_units'] * self.hyperparameters['nn_unit_decay'] ** layer_num)

            layer = torch.nn.Linear(input_dim, output_dim)

            model_layers = self.add_layer_and_relu(layer, model_layers)

            input_dim = output_dim

        output_dim = self.action_size
        layer = torch.nn.Linear(input_dim, output_dim)

        model_layers.append(layer)

        return model_layers

    def add_layer_and_relu(self, layer, model_layers):
        model_layers.append(layer)
        model_layers.append(torch.nn.ReLU())
        return model_layers



    def get_model(self):
        return self.model


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