import torch
import numpy as np
from torch import optim
from torch.distributions import Categorical

from Agents.Base_Agent import Base_Agent
from Model import Model
from Policy_Gradient_Agents.REINFORCE_Agent import REINFORCE_Agent


""" WIP NOT FINISHED YET"""

# TODO implement parallelism
# TODO calculate advantages rather than just discounted return


class PPO_Agent(Base_Agent):
    agent_name = "PPO"

    def __init__(self, config):

        Base_Agent.__init__(self, config)

        self.policy_new = Model(self.state_size, self.action_size, config.seed, self.hyperparameters).to(self.device)
        self.policy_old = Model(self.state_size, self.action_size, config.seed, self.hyperparameters).to(self.device)
        self.max_steps_per_episode = config.environment.get_max_steps_per_episode()
        self.policy_new_optimizer = optim.Adam(self.policy_new.parameters(), lr=self.hyperparameters["learning_rate"])
        self.episode_number = 0

        self.many_episode_states = []
        self.many_episode_actions = []
        self.many_episode_rewards = []

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.environment.reset_environment()
        self.state = self.environment.get_state()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.one_episode_states = []
        self.one_episode_actions = []
        self.one_episode_rewards = []
        self.episode_policy_probability_ratios = []
        self.episode_step_number = 0

    def step(self):
        """Runs a step within a game including a learning step if required"""
        self.pick_and_conduct_action()

        self.update_next_state_reward_done_and_score()
        self.store_state_action_and_reward()


        if self.done:
            self.many_episode_states.append(self.one_episode_states)
            self.many_episode_actions.append(self.one_episode_actions)
            self.many_episode_rewards.append(self.one_episode_rewards)

            if self.time_to_learn():
                for _ in range(self.hyperparameters["learning_iterations_per_round"]):
                    self.policy_learn()

                self.many_episode_states = []
                self.many_episode_actions = []
                self.many_episode_rewards = []
                self.equalise_policies()

        self.state = self.next_state #this is to set the state for the next iteration

    def pick_and_conduct_action(self):
        self.action = self.pick_action()
        self.conduct_action()

    def pick_action(self):
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        new_policy_action_probabilities = self.policy_new.forward(state).cpu()
        action_distribution = Categorical(new_policy_action_probabilities) # this creates a distribution to sample from
        action = action_distribution.sample().numpy()[0]
        return action

    def store_state_action_and_reward(self):
        self.one_episode_states.append(self.state)
        self.one_episode_actions.append(self.action)
        self.one_episode_rewards.append(self.reward)

    def time_to_learn(self):
        return self.episode_number % self.hyperparameters["episodes_per_learning_round"] == 0

    def policy_learn(self):

        all_ratio_of_policy_probabilities = []
        all_discounted_returns = []


        for episode in range(self.hyperparameters["episodes_per_learning_round"]):

            states = self.many_episode_states[episode]
            actions = self.many_episode_actions[episode]
            rewards = self.many_episode_rewards[episode]

            for ix in range(len(states)):
                discounted_return = self.calculate_discounted_reward(rewards[ix:])
                all_discounted_returns.append(discounted_return)

                ratio_of_policy_probabilities = self.calculate_policy_action_probability_ratio(states, actions, ix)
                all_ratio_of_policy_probabilities.append(ratio_of_policy_probabilities)


        loss = self.calculate_loss(all_ratio_of_policy_probabilities, all_discounted_returns)

        self.take_policy_new_optimisation_step(loss)


    def calculate_discounted_reward(self, rewards):
        discounts = self.hyperparameters["discount_rate"] ** np.arange(len(rewards))
        total_discounted_reward = np.dot(discounts, rewards)
        return total_discounted_reward

    def calculate_policy_action_probability_ratio(self, states, actions, ix):

        state = states[ix]
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        new_policy_action_probabilities = self.policy_new.forward(state).cpu()[0]
        old_policy_action_probabilities = self.policy_old.forward(state).cpu()[0]

        action_chosen = actions[ix]

        ratio_of_policy_probabilities = new_policy_action_probabilities[action_chosen] / \
                                        old_policy_action_probabilities[action_chosen]

        return ratio_of_policy_probabilities

    def calculate_loss(self, all_ratio_of_policy_probabilities, all_discounted_returns):

        loss = 0

        for policy_probability_ratio, discounted_return in zip(all_ratio_of_policy_probabilities, all_discounted_returns):
            clipped_probability_ratio = torch.clamp(policy_probability_ratio, min=1.0 - self.hyperparameters["clip_epsilon"],
                                                    max=1.0 + self.hyperparameters["clip_epsilon"])
            observation_loss = torch.min(policy_probability_ratio * discounted_return, clipped_probability_ratio * discounted_return)
            loss -= observation_loss

        average_loss = loss / len(all_ratio_of_policy_probabilities)

        return average_loss

    def take_policy_new_optimisation_step(self, loss):
        self.policy_new.zero_grad()  # reset gradients to 0
        loss.backward()  # this calculates the gradients
        self.policy_new_optimizer.step()  # this applies the gradients

    def equalise_policies(self):
        for old_param, new_param in zip(self.policy_old.parameters(), self.policy_new.parameters()):
            old_param.data.copy_(new_param.data)
