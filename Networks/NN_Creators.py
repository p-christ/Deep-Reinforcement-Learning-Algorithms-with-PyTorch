import torch
from torch import nn
from torch.nn.init import xavier_normal_



def create_vanilla_NN(state_size, action_size, seed, hyperparameters):

    torch.manual_seed(seed)
    model_layers = create_model_layers(state_size, action_size, hyperparameters)

    model = torch.nn.Sequential(*model_layers)
    model.apply(linear_layer_weights_xavier_initialisation)

    return model

def create_model_layers(state_size, action_size, hyperparameters):
    model_layers = []

    input_dim = state_size

    for layer_num in range(hyperparameters["nn_layers"] - 1):
        output_dim = int(hyperparameters['nn_start_units'] * hyperparameters['nn_unit_decay'] ** layer_num)

        add_linear_layer(input_dim, output_dim, model_layers)

        if hyperparameters["batch_norm"]:
            add_batch_norm_layer(output_dim, model_layers)

        add_relu_layer(model_layers)

        input_dim = output_dim

    output_dim = action_size

    add_linear_layer(input_dim, output_dim, model_layers)

    if hyperparameters["softmax_final_layer"]:
        add_softmax_layer(model_layers)

    return model_layers

def add_linear_layer(input_dim, output_dim, model_layers):
    layer = torch.nn.Linear(input_dim, output_dim)
    model_layers.append(layer)

def add_batch_norm_layer(output_dim, model_layers):
    layer = torch.nn.BatchNorm1d(output_dim)
    model_layers.append(layer)

def add_relu_layer(model_layers):
    model_layers.append(torch.nn.ReLU())

def add_softmax_layer(model_layers):
    model_layers.append(torch.nn.Softmax())



def linear_layer_weights_xavier_initialisation(layer):
    if isinstance(layer, nn.Linear):
        xavier_normal_(layer.weight.data)
        # xavier_normal_(layer.bias.data)
