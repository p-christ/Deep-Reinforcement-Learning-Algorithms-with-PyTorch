import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_


""" WIP - not complete """

class Model(nn.Module):
    """Creates the neural network described by the hyperparameters you provide"""

    def __init__(self, state_size, action_size, seed, hyperparameters):
        nn.Module.__init__(self)
        self.model = self.create_vanilla_NN(state_size, action_size, seed, hyperparameters)

    def forward(self, input):
        return self.model(input)

    def create_vanilla_NN(self, state_size, action_size, seed, hyperparameters):
        """Creates a neural network model"""
        torch.manual_seed(seed)
        model_layers = self.create_model_layers(state_size, action_size, hyperparameters)
        model = torch.nn.Sequential(*model_layers)
        model.apply(self.linear_layer_weights_xavier_initialisation)
        return model

    def create_model_layers(self, state_size, action_size, hyperparameters):
        """Creates the different layers of a vanilla neural network"""
        model_layers = []

        input_dim = state_size

        for layer_num in range(hyperparameters["nn_layers"] - 1):
            output_dim = int(hyperparameters['nn_start_units'] * hyperparameters['nn_unit_decay'] ** layer_num)

            self.add_linear_layer(input_dim, output_dim, model_layers)

            if hyperparameters["batch_norm"]:
                self.add_batch_norm_layer(output_dim, model_layers)

            self.add_relu_layer(model_layers)

            input_dim = output_dim

        output_dim = action_size

        self.add_linear_layer(input_dim, output_dim, model_layers)

        if hyperparameters["final_layer_activation"] == "SOFTMAX":
            self.add_softmax_layer(model_layers)

        if hyperparameters["final_layer_activation"] == "TANH":
            self.add_tanh_layer(model_layers)

        return model_layers

    def add_linear_layer(self, input_dim, output_dim, model_layers):
        layer = torch.nn.Linear(input_dim, output_dim)
        model_layers.append(layer)

    def add_batch_norm_layer(self, output_dim, model_layers):
        layer = torch.nn.BatchNorm1d(output_dim)
        model_layers.append(layer)

    def add_relu_layer(self, model_layers):
        model_layers.append(torch.nn.ReLU())

    def add_softmax_layer(self, model_layers):
        model_layers.append(torch.nn.Softmax(dim=1))

    def add_tanh_layer(self, model_layers):
        model_layers.append(torch.nn.Tanh())

    def linear_layer_weights_xavier_initialisation(self, layer):
        if isinstance(layer, nn.Linear):
            xavier_normal_(layer.weight.data)