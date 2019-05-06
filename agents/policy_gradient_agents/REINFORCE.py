import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from agents.Base_Agent import Base_Agent

class REINFORCE(Base_Agent):
    agent_name = "REINFORCE"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.policy = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.hyperparameters["learning_rate"])
        self.episode_rewards = []
        self.episode_log_probabilities = []

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.state = self.environment.reset_environment()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_rewards = []
        self.episode_log_probabilities = []
        self.episode_step_number = 0

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.pick_and_conduct_action_and_save_log_probabilities()
            self.update_next_state_reward_done_and_score()
            self.store_reward()
            if self.time_to_learn():
                self.actor_learn()
            self.state = self.next_state #this is to set the state for the next iteration
            self.episode_step_number += 1
        self.episode_number += 1

    def pick_and_conduct_action_and_save_log_probabilities(self):
        """Picks and then conducts actions. Then saves the log probabilities of the actions it conducted to be used for
        learning later"""
        action, log_probabilities = self.pick_action_and_get_log_probabilities()
        self.store_log_probabilities(log_probabilities)
        self.store_action(action)
        self.conduct_action()

    def pick_action_and_get_log_probabilities(self):
        """Picks actions and then calculates the log probabilities of the actions it picked given the policy"""
        # PyTorch only accepts mini-batches and not individual observations so we have to add
        # a "fake" dimension to our observation using unsqueeze
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        action_probabilities = self.policy.forward(state).cpu()
        action_distribution = Categorical(action_probabilities) # this creates a distribution to sample from
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)

    def store_log_probabilities(self, log_probabilities):
        """Stores the log probabilities of picked actions to be used for learning later"""
        self.episode_log_probabilities.append(log_probabilities)

    def store_action(self, action):
        """Stores the action picked"""
        self.action = action

    def store_reward(self):
        """Stores the reward picked"""
        self.episode_rewards.append(self.reward)

    def actor_learn(self):
        """Runs a learning iteration for the policy"""
        total_discounted_reward = self.calculate_episode_discounted_reward()
        policy_loss = self.calculate_policy_loss_on_episode(total_discounted_reward)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def calculate_episode_discounted_reward(self):
        """Calculates the cumulative discounted return for the episode"""
        discounts = self.hyperparameters["discount_rate"] ** np.arange(len(self.episode_rewards))
        total_discounted_reward = np.dot(discounts, self.episode_rewards)
        return total_discounted_reward

    def calculate_policy_loss_on_episode(self, total_discounted_reward):
        """Calculates the loss from an episode"""
        policy_loss = []
        for log_prob in self.episode_log_probabilities:
            policy_loss.append(-log_prob * total_discounted_reward)
        policy_loss = torch.cat(policy_loss).sum() # We need to add up the losses across the mini-batch to get 1 overall loss
        return policy_loss

    def time_to_learn(self):
        """Tells us whether it is time for the algorithm to learn. With REINFORCE we only learn at the end of every
        episode so this just returns whether the episode is over"""
        return self.done
