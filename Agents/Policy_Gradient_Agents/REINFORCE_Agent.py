import random
import numpy as np
import torch
from torch.distributions import Categorical
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Base_Agent import Base_Agent
from NN_Creators import create_vanilla_NN


class REINFORCE_Agent(Base_Agent):

    def __init__(self, environment, seed, hyperparameters, rolling_score_length, average_score_required,
                 agent_name):

        hyperparameters = hyperparameters["Policy_Gradient_Agents"]

        Base_Agent.__init__(self, environment=environment,
                            seed=seed, hyperparameters=hyperparameters, rolling_score_length=rolling_score_length,
                            average_score_required=average_score_required, agent_name=agent_name)


        self.policy = create_vanilla_NN(self.state_size, self.action_size, seed, self.hyperparameters).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.hyperparameters["learning_rate"])
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'max'
        # Manual way: ? ?
        # for g in optim.param_groups:
        #     g['lr'] = 0.001)

        self.episode_rewards = []
        self.episode_log_probabilities = []

    def step(self):
        """Runs a step within a game including a learning step if required"""
        self.pick_and_conduct_action_and_save_log_probabilities()

        self.update_next_state_reward_done_and_score()
        self.store_reward()

        if self.time_to_learn():
            self.learn()

        self.save_experience()
        self.state = self.next_state #this is to set the state for the next iteration


    def pick_and_conduct_action_and_save_log_probabilities(self):

        action, log_probabilities = self.pick_action_and_get_log_probabilities()

        self.store_log_probabilities(log_probabilities)
        self.store_action(action)

        self.conduct_action()

    def store_reward(self):
        self.episode_rewards.append(self.reward)

    def store_action(self, action):
        self.action = action

    def pick_action_and_get_log_probabilities(self):

        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)

        action_probabilities = self.policy.forward(state).cpu()
        # action_probabilities = torch.nn.functional.softmax(action_probabilities) # we could put this in network class instead

        action_distribution = Categorical(action_probabilities) # this creates a distribution to sample from
        action = action_distribution.sample()


        return action.item(), action_distribution.log_prob(action)

    def store_log_probabilities(self, log_probabilities):
        self.episode_log_probabilities.append(log_probabilities)


    def learn(self):

        total_discounted_reward = self.calculate_episode_discounted_reward()
        policy_loss = self.calculate_policy_loss_on_episode(total_discounted_reward)

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def calculate_episode_discounted_reward(self):
        discounts = self.hyperparameters["discount_rate"] ** np.arange(len(self.episode_rewards))
        total_discounted_reward = np.dot(discounts, self.episode_rewards)

        return total_discounted_reward

    def calculate_policy_loss_on_episode(self, total_discounted_reward):
        policy_loss = []
        for log_prob in self.episode_log_probabilities:
            policy_loss.append(-log_prob * total_discounted_reward)
        policy_loss = torch.cat(policy_loss).sum() #This concatenates the sequence and then adds them up
        # policy_loss = Variable(policy_loss, requires_grad = True)
        return policy_loss


    def time_to_learn(self):
        """With REINFORCE we only learn at the end of every episode"""
        return self.done


    def save_experience(self):
        """We don't save our experiences with this algorithm"""
        pass


    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.environment.reset_environment()
        self.state = self.environment.get_state()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.score = 0
        self.episode_rewards = []
        self.episode_log_probabilities = []




    def locally_save_policy(self):
        torch.save(self.qnetwork_local.state_dict(), "Models/{}_policy.pt".format(self.agent_name))

