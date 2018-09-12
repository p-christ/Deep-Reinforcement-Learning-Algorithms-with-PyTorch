import torch
import torch.nn as nn
import torch.nn.functional as F



def create_vanilla_NN(state_size, action_size, seed, hyperparameters):

    torch.manual_seed(seed)
    model_layers = create_model_layers(state_size, action_size, hyperparameters)

    model = torch.nn.Sequential(*model_layers)
    return model

def create_model_layers(state_size, action_size, hyperparameters):
    model_layers = []

    input_dim = state_size

    for layer_num in range(hyperparameters["nn_layers"] - 1):
        output_dim = int(hyperparameters['nn_start_units'] * hyperparameters['nn_unit_decay'] ** layer_num)

        add_linear_layer(input_dim, output_dim, model_layers)
        add_relu_layer(model_layers)

        input_dim = output_dim

    output_dim = action_size

    add_linear_layer(input_dim, output_dim, model_layers)

    return model_layers

def add_linear_layer(input_dim, output_dim, model_layers):
    layer = torch.nn.Linear(input_dim, output_dim)
    model_layers.append(layer)

def add_relu_layer(model_layers):
    model_layers.append(torch.nn.ReLU())

