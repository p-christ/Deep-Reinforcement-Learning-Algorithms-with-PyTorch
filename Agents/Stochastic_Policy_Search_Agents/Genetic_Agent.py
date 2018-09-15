import copy
import random

import torch
import numpy as np

from Base_Agent import Base_Agent
from Linear_Model import Linear_Model

class Genetic_Agent(Base_Agent):

    def __init__(self, environment, seed, hyperparameters, rolling_score_length, average_score_required,
                 agent_name):

        self.hyperparameters = hyperparameters["Stochastic_Policy_Search_Agents"]

        Base_Agent.__init__(self, environment=environment,
                            seed=seed, hyperparameters=self.hyperparameters, rolling_score_length=rolling_score_length,
                            average_score_required=average_score_required, agent_name=agent_name)

        self.num_policies = self.hyperparameters["num_policies"]

        if self.hyperparameters["policy_network_type"] == "Linear":
            self.policies = [Linear_Model(self.state_size, self.action_size) for _ in range(self.num_policies)]

            self.weight_rows = self.policies[0].weights.shape[0]
            self.weight_cols = self.policies[0].weights.shape[1]

        self.stochastic_action_decision = self.hyperparameters["stochastic_action_decision"]

        self.policy_to_use_this_episode = 0
        self.policy_scores_this_round = [0] * self.num_policies

        self.policy_that_achieves_score = None
        self.found_policy_that_achieves_score = None

    def step(self):
        """Runs a step within a game including a learning step if required"""
        self.pick_and_conduct_action()

        self.update_next_state_reward_done_and_score()

        if self.time_to_learn():
            self.learn()

        self.save_experience()
        self.state = self.next_state #this is to set the state for the next iteration

        if self.done:

            if not self.found_policy_that_achieves_score:
                self.policy_scores_this_round[self.policy_to_use_this_episode] += self.total_episode_score_so_far

                # if self.total_episode_score_so_far >= self.average_score_required:
                #     self.found_policy_that_achieves_score = True


                if not self.found_policy_that_achieves_score:
                    self.update_policy_to_use_this_episode()
            #
            # if self.found_policy_that_achieves_score:
            #     print(self.total_episode_score_so_far)

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

        if self.policy_to_use_this_episode < self.num_policies - 1:
            self.policy_to_use_this_episode += 1
        else:
            self.policy_to_use_this_episode = 0

    def time_to_learn(self):
        return self.episode_number % (self.num_policies * 10) == 0

    def save_experience(self):
        """We don't save past experiences for this algorithm"""
        pass

    def learn(self):

        if not self.found_policy_that_achieves_score:


            policy_ranks = list(reversed(np.argsort(self.policy_scores_this_round)))

            elite_set_of_policies = [self.policies[x] for x in policy_ranks[:5]]

            print(self.policy_scores_this_round)

            self.policy_scores_this_round = [x / 10.0 for x in self.policy_scores_this_round]

            print("H")
            print(self.policy_scores_this_round)
            print("end")

            if max(self.policy_scores_this_round) >= self.average_score_required:
                self.found_policy_that_achieves_score = True
                print(self.policy_scores_this_round)
                pass

            select_probs = np.array(self.policy_scores_this_round) / np.sum(self.policy_scores_this_round)

            child_policy_set = [self.crossover(
                self.policies[np.random.choice(range(self.num_policies), p=select_probs)],
                self.policies[np.random.choice(range(self.num_policies), p=select_probs)])
                for _ in range(self.num_policies - 5)]
            mutated_policies = [self.mutation(policy) for policy in child_policy_set]

            self.policies = elite_set_of_policies + mutated_policies

            self.policy_scores_this_round = [0] * self.num_policies



    def crossover(self, policy1, policy2):
        new_policy = copy.deepcopy(policy1)

        for row in range(self.weight_rows):
            for col in range(self.weight_cols):
                rand = np.random.uniform()
                if rand > 0.5:
                    new_policy.weights[row][col] = policy2.weights[row][col]
        return new_policy

    def mutation(self, policy, p=0.05):
        new_policy = copy.deepcopy(policy)
        for row in range(self.weight_rows):
            for col in range(self.weight_cols):
                rand = np.random.uniform()
                if rand < p:
                    new_policy.weights[row][col] = 2.0 * (random.random() - 0.5)
        return new_policy

    def locally_save_policy(self):
        pass