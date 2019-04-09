import numpy as np
from abc import ABCMeta
import torch
from torch.distributions import Categorical, normal, MultivariateNormal

def abstract(cls):
    return ABCMeta(cls.__name__, cls.__bases__, dict(cls.__dict__))
    
def save_score_results(file_path, results):
    """Saves results as a numpy file at given path"""
    np.save(file_path, results)

def normalise_rewards(rewards):
    """Normalises rewards to mean 0 and standard deviation 1"""
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return (rewards - mean_reward) / (std_reward + 1e-8) #1e-8 added for stability

def create_actor_distribution(action_types, actor_output, action_size):
    """Creates a distribution that the actor can then use to randomly draw actions"""
    if action_types == "DISCRETE":
        assert actor_output.size()[1] == action_size, "Actor output the wrong size"
        action_distribution = Categorical(actor_output)  # this creates a distribution to sample from
    else:
        assert actor_output.size()[1] == action_size * 2, "Actor output the wrong size"
        means = actor_output[:, :action_size].squeeze(0)
        stds = actor_output[:,  action_size:].squeeze(0)
        if len(means.shape) == 2: means = means.squeeze(-1)
        if len(stds.shape) == 2: stds = stds.squeeze(-1)
        if len(stds.shape) > 1 or len(means.shape) > 1: raise ValueError("Wrong mean and std shapes")
        action_distribution = normal.Normal(means.squeeze(0), torch.abs(stds))
    return action_distribution

