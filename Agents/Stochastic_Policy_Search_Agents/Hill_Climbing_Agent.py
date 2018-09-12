import copy

import torch

from Base_Agent import Base_Agent
from Linear_Model import Linear_Model

import numpy as np



class Hill_Climbing_Agent(Base_Agent):

    def __init__(self, environment, seed, hyperparameters, rolling_score_length, average_score_required,
                 agent_name):


        Base_Agent.__init__(self, environment=environment,
                            seed=seed, hyperparameters=hyperparameters, rolling_score_length=rolling_score_length,
                            average_score_required=average_score_required, agent_name=agent_name)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.hyperparameters["policy_network_type"] == "Linear":
            self.policy = Linear_Model(self.state_size, self.action_size)
            self.best_weights_seen = self.policy.weights

        self.best_episode_score_seen = float("-inf")

        self.noise_scale = 1e-2

    def time_to_learn(self):
        """Tells agent to perturb weights at end of every episode"""
        return self.done

    def pick_action(self):

        policy_values = self.policy.forward(self.state)
        action = np.argmax(policy_values)

        return action


    def learn(self):

        raw_noise = (2.0*(np.random.rand(*self.policy.weights.shape) - 0.5))

        if self.score >= self.best_episode_score_seen:

            self.best_episode_score_seen = self.score
            self.best_weights_seen = self.policy.weights
            noise_scale = max(1e-3, self.noise_scale / 2.0)
            self.policy.weights += noise_scale * raw_noise

        else:

            noise_scale = min(2.0, self.noise_scale * 2.0)
            self.policy.weights = self.best_weights_seen + noise_scale * raw_noise

    def save_experience(self):
        pass



    def locally_save_policy(self):
        pass

