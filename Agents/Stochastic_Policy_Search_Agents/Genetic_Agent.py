import copy
import random

import numpy as np

from Agents.Base_Agent import Base_Agent
from Utilities.Models.Linear_Model import Linear_Model

class Genetic_Agent(Base_Agent):
    agent_name = "Genetic_Agent"

    def __init__(self, config):

        Base_Agent.__init__(self, config)

        self.num_policies = self.hyperparameters["num_policies"]

        if self.hyperparameters["policy_network_type"] == "Linear":
            self.policies = [Linear_Model(self.state_size, self.action_size) for _ in range(self.num_policies)]

            self.weight_rows = self.policies[0].weights.shape[0]
            self.weight_cols = self.policies[0].weights.shape[1]

        self.stochastic_action_decision = self.hyperparameters["stochastic_action_decision"]
        self.episodes_per_policy = self.hyperparameters["episodes_per_policy"]
        self.num_policies_to_keep = self.hyperparameters["num_policies_to_keep"]

        self.policy_to_use_this_episode = 0
        self.policy_scores_this_round = [0] * self.num_policies

    def step(self):
        """Runs a step within a game including a learning step if required"""
        self.episode_number += 1
        while not self.done:
            self.pick_and_conduct_action()
            self.update_next_state_reward_done_and_score()
            if self.time_to_learn():
                self.critic_learn()
            self.state = self.next_state #this is to set the state for the next iteration

            if self.done:
                self.policy_scores_this_round[self.policy_to_use_this_episode] += self.total_episode_score_so_far
                self.update_policy_to_use_this_episode()
            self.episode_step_number += 1

    def pick_and_conduct_action(self):
        self.action = self.pick_action()
        self.conduct_action()

    def pick_action(self):
        policy_values = self.policies[self.policy_to_use_this_episode].forward(self.state)
        if self.stochastic_action_decision:
            action = np.random.choice(self.action_size, p=policy_values) # option 1: stochastic policy
        else:
            action = np.argmax(policy_values)  # option 2: deterministic policy
        return action

    def update_policy_to_use_this_episode(self):
        """Sets which policy to use for the next episode"""
        if self.policy_to_use_this_episode < self.num_policies - 1:
            self.policy_to_use_this_episode += 1
        else:
            self.policy_to_use_this_episode = 0

    def time_to_learn(self):
        """Tells us if it is time to learn"""
        return self.done and self.episode_number % (self.num_policies * self.episodes_per_policy) == 0

    def critic_learn(self):
        """Creates a new set of policies by evolving the previous policies and resets scores"""
        self.policies = self.create_new_set_of_policies()
        self.reset_round_policy_scores()

    def create_new_set_of_policies(self):
        """Creates the new set of policies by evolving the previous policies"""
        policies_in_score_order = list(np.argsort(self.policy_scores_this_round))

        elite_set_of_policies = [self.policies[policy_index] for policy_index in policies_in_score_order[-1 * self.num_policies_to_keep:]]
        self.policy_scores_this_round = [x / self.episodes_per_policy for x in self.policy_scores_this_round]
        policy_selection_probabilities = np.array(self.policy_scores_this_round) / np.sum(self.policy_scores_this_round)
        child_policy_set = self.create_child_policies(policy_selection_probabilities)
        mutated_policies = [self.mutation(policy) for policy in child_policy_set]

        return elite_set_of_policies + mutated_policies

    def create_child_policies(self, policy_selection_probabilities):
        """Creates a set of policies that are a crossover of the previous policies. Each policy is chosen
        with a probability proportional to the score they achieved this round"""
        child_policy_set = [self.crossover(
            self.policies[np.random.choice(range(self.num_policies), p=policy_selection_probabilities)],
            self.policies[np.random.choice(range(self.num_policies), p=policy_selection_probabilities)])
            for _ in range(self.num_policies - self.num_policies_to_keep)]
        return child_policy_set

    def reset_round_policy_scores(self):
        self.policy_scores_this_round = [0] * self.num_policies

    def crossover(self, policy1, policy2):
        """Mixes together the two policies. Each policy weight has a 50:50 chance of coming from
         policy1 or policy2"""
        new_policy = copy.deepcopy(policy1)

        for row in range(self.weight_rows):
            for col in range(self.weight_cols):
                rand = np.random.uniform()
                if rand > 0.5:
                    new_policy.weights[row][col] = policy2.weights[row][col]
        return new_policy

    # TODO - just create a matrix of random values and add that in one go rather than 1 value by 1
    def mutation(self, policy, p=0.05):
        """For each weight in the policy, with some probability this replaces the weight with a random number"""
        new_policy = copy.deepcopy(policy)
        for row in range(self.weight_rows):
            for col in range(self.weight_cols):
                rand = np.random.uniform()
                if rand < p:
                    new_policy.weights[row][col] = 2.0 * (random.random() - 0.5)
        return new_policy