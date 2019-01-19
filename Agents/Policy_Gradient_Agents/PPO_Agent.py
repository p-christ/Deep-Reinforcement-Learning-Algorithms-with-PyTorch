import sys
import torch
import numpy as np
from torch import optim
from Agents.Base_Agent import Base_Agent
from Utilities.Models.Neural_Network import Neural_Network
from Utilities.Parallel_Experience_Generator import Parallel_Experience_Generator
from Utilities.Utility_Functions import normalise_rewards, create_actor_distribution

class PPO_Agent(Base_Agent):
    agent_name = "PPO"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.policy_output_size = self.calculate_policy_output_size()
        self.policy_new = Neural_Network(self.state_size, self.policy_output_size, self.random_seed, self.hyperparameters,
                                         "VANILLA_NN").to(self.device)
        self.policy_old = Neural_Network(self.state_size, self.policy_output_size, self.random_seed, self.hyperparameters,
                                         "VANILLA_NN").to(self.device)
        self.max_steps_per_episode = config.environment.get_max_steps_per_episode()
        self.policy_new_optimizer = optim.Adam(self.policy_new.parameters(), lr=self.hyperparameters["learning_rate"])
        self.episode_number = 0
        self.many_episode_states = []
        self.many_episode_actions = []
        self.many_episode_rewards = []
        self.experience_generator = Parallel_Experience_Generator(self.environment, self.policy_new, self.random_seed, self.hyperparameters)

    def calculate_policy_output_size(self):
        """Initialises the policies"""
        if self.action_types == "DISCRETE":
            return self.action_size
        elif self.action_types == "CONTINUOUS":
            return self.action_size * 2 #Because we need 1 parameter for mean and 1 for std of distribution

    def step(self):
        self.many_episode_states, self.many_episode_actions, self.many_episode_rewards = self.experience_generator.play_n_episodes(
            self.hyperparameters["episodes_per_learning_round"])
        self.episode_number += self.hyperparameters["episodes_per_learning_round"]
        self.policy_learn()
        self.update_learning_rate(self.hyperparameters["learning_rate"], self.policy_new_optimizer)
        self.equalise_policies()

    def policy_learn(self):
        """A learning round for the policy"""
        all_discounted_returns = self.calculate_all_discounted_returns()
        if self.hyperparameters["normalise_rewards"]:
            all_discounted_returns = normalise_rewards(all_discounted_returns)
        for _ in range(self.hyperparameters["learning_iterations_per_round"]):
            all_ratio_of_policy_probabilities = self.calculate_all_ratio_of_policy_probabilities()
            loss = self.calculate_loss([all_ratio_of_policy_probabilities], all_discounted_returns)
            self.take_policy_new_optimisation_step(loss)

    def calculate_all_discounted_returns(self):
        all_discounted_returns = []
        for episode in range(len(self.many_episode_states)):
            discounted_returns = [0]
            for ix in range(len(self.many_episode_states[episode])):
                return_value = self.many_episode_rewards[episode][-(ix + 1)] + self.hyperparameters["discount_rate"]*discounted_returns[-1]
                discounted_returns.append(return_value)
            discounted_returns = discounted_returns[1:]
            all_discounted_returns.extend(discounted_returns[::-1])
        return all_discounted_returns

    def calculate_all_ratio_of_policy_probabilities(self):

        all_states = [state for states in self.many_episode_states for state in states]
        all_actions = [action for actions in self.many_episode_actions for action in actions]

        all_states = torch.stack([torch.Tensor(states).float().to(self.device) for states in all_states])
        all_actions = torch.stack([torch.Tensor(actions).float().to(self.device) for actions in all_actions])
        all_actions = all_actions.view(-1, len(all_states))

        new_policy_distribution_log_prob = self.calculate_log_probability_of_actions(self.policy_new, all_states, all_actions)
        old_policy_distribution_log_prob = self.calculate_log_probability_of_actions(self.policy_old, all_states, all_actions)
        ratio_of_policy_probabilities = torch.exp(new_policy_distribution_log_prob) / (torch.exp(old_policy_distribution_log_prob) + 1e-8)
        return ratio_of_policy_probabilities

    def calculate_log_probability_of_actions(self, policy, states, actions):
        """Calculates the log probability of an action occuring given a policy and starting state"""
        policy_output = policy.forward(states).to(self.device)
        policy_distribution = create_actor_distribution(self.action_types, policy_output, self.action_size)
        actions_tensor = actions
        policy_distribution_log_prob = policy_distribution.log_prob(actions_tensor)
        return policy_distribution_log_prob

    def calculate_loss(self, all_ratio_of_policy_probabilities, all_discounted_returns):
        """Calculates the PPO loss"""
        all_ratio_of_policy_probabilities = torch.squeeze(torch.stack(all_ratio_of_policy_probabilities))
        all_ratio_of_policy_probabilities = torch.clamp(input=all_ratio_of_policy_probabilities,
                                                        min = -sys.maxsize,
                                                        max = sys.maxsize)
        all_discounted_returns = torch.tensor(all_discounted_returns).to(all_ratio_of_policy_probabilities)
        potential_loss_value_1 = all_discounted_returns * all_ratio_of_policy_probabilities
        potential_loss_value_2 = all_discounted_returns * self.clamp_probability_ratio(all_ratio_of_policy_probabilities)
        loss = torch.min(potential_loss_value_1, potential_loss_value_2)
        loss = -torch.mean(loss)
        return loss

    def clamp_probability_ratio(self, value):
        """Clamps a value between a certain range determined by hyperparameter clip epsilon"""
        return torch.clamp(input=value, min=1.0 - self.hyperparameters["clip_epsilon"],
                                  max=1.0 + self.hyperparameters["clip_epsilon"])

    def take_policy_new_optimisation_step(self, loss):
        self.policy_new_optimizer.zero_grad()  # reset gradients to 0
        loss.backward()  # this calculates the gradients
        torch.nn.utils.clip_grad_norm_(self.policy_new.parameters(), self.hyperparameters[
            "gradient_clipping_norm"])  # clip gradients to help stabilise training
        self.policy_new_optimizer.step()  # this applies the gradients

    def equalise_policies(self):
        """Sets the old policy's parameters equal to the new policy's parameters"""
        for old_param, new_param in zip(self.policy_old.parameters(), self.policy_new.parameters()):
            old_param.data.copy_(new_param.data)

    def save_result(self):
        for ep in range(len(self.many_episode_rewards)):
            total_reward = np.sum(self.many_episode_rewards[ep])
            self.game_full_episode_scores.append(total_reward)
            self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
        self.save_max_result_seen()