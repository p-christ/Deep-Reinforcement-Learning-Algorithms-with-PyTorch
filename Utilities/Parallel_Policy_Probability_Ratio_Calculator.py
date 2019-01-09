import torch

from Utility_Functions import create_actor_distribution
import numpy as np


class Parallel_Policy_Probability_Ratio_Calculator(object):

    def __init__(self, policy_new, policy_old, action_types, action_size):

        self.action_types = action_types
        self.action_size = action_size

        self.policy_new = policy_new
        self.policy_old = policy_old

    def calculate_all_ratio_of_policy_probabilities(self):



    all_ratio_of_policy_probabilities = []
    for episode in range(len(self.many_episode_states)):
        for ix in range(len(self.many_episode_states[episode])):
            ratio_of_policy_probabilities = self.calculate_policy_action_probability_ratio(
                self.many_episode_states[episode][ix], self.many_episode_actions[episode][ix])
            all_ratio_of_policy_probabilities.append(ratio_of_policy_probabilities)


    def calculate_policy_action_probability_ratio(self, state, action):
        """Calculates the ratio of the probability/density of an action wrt the old and new policy"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        new_policy_distribution_log_prob = self.calculate_log_probability_of_action(self.policy_new, state, action)
        old_policy_distribution_log_prob = self.calculate_log_probability_of_action(self.policy_old, state, action)
        ratio_of_policy_probabilities = torch.exp(new_policy_distribution_log_prob) / torch.exp(old_policy_distribution_log_prob)
        return ratio_of_policy_probabilities

    def calculate_log_probability_of_action(self, policy, state, action):
        """Calculates the log probability of an action occuring given a policy and starting state"""
        policy_output = policy.forward(state).cpu()
        policy_distribution = create_actor_distribution(self.action_types, policy_output, self.action_size)
        policy_distribution_log_prob = policy_distribution.log_prob(torch.from_numpy(np.array(action)))
        return policy_distribution_log_prob
