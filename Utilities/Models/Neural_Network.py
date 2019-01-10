import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_


""" WIP - not complete """

# TODO add batch normalisation


class Neural_Network(nn.Module):
    """Creates the neural network described by the hyperparameters you provide"""

    def __init__(self, state_size, action_size, random_seed, hyperparameters, model_type):
        nn.Module.__init__(self)
        self.hyperparameters = hyperparameters
        self.model_type = model_type
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random_seed
        self.vanilla_model = self.create_vanilla_NN()


    def forward(self, input):

        if self.model_type == "VANILLA_NN":
            return self.vanilla_model(input)

        if self.model_type == "DUELLING_NN":
            return self.duelling_model(input)

    def create_vanilla_NN(self):
        """Creates a neural network model"""
        torch.manual_seed(self.random_seed)
        model_layers, input_dim_to_next_layer = self.create_intemediary_model_layers()
        final_layer = self.create_final_layer(input_dim_to_next_layer)
        model_layers.extend(final_layer)
        model = torch.nn.Sequential(*model_layers)
        model.apply(self.linear_layer_weights_xavier_initialisation)
        return model

    def create_intemediary_model_layers(self):
        """Creates the different layers of a vanilla neural network"""
        model_layers = []

        input_dim = self.state_size

        for layer_num in range(self.hyperparameters["nn_layers"] - 1):
            output_dim = int(self.hyperparameters['nn_start_units'] * self.hyperparameters['nn_unit_decay'] ** layer_num)
            self.add_linear_layer(input_dim, output_dim, model_layers)
            if self.hyperparameters["batch_norm"]:
                self.add_batch_norm_layer(output_dim, model_layers)
            self.add_relu_layer(model_layers)
            input_dim = output_dim

        return model_layers, input_dim

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

    def create_final_layer(self, input_dim_to_next_layer):

        final_layer = []
        self.add_linear_layer(input_dim_to_next_layer, self.action_size, final_layer)
        if self.hyperparameters["final_layer_activation"] == "SOFTMAX":
            self.add_softmax_layer(final_layer)
        if self.hyperparameters["final_layer_activation"] == "TANH":
            self.add_tanh_layer(final_layer)
        return final_layer

    def duelling_model(self, input):
        raise ValueError("Duelling network architecture not implemented yet")