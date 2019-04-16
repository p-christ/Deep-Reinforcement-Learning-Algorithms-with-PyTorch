import copy
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from Base_Agent import Base_Agent
from contextlib import closing
from multiprocessing import Pool
from torch.multiprocessing import Pool as GPU_POOL
from PPO import PPO
from Parallel_Experience_Generator import Parallel_Experience_Generator
from Utility_Functions import create_actor_distribution

# mp.Queue a FIFO queue
# mp.Process
# Pytorch has its own multiprocessing
# https://www.youtube.com/watch?v=O5BlozCJBSE&t=207s
# process_count = mp.cpu_count()
# total_envs = 64
# envs_per_process = 64 / process_count

class A2C(Base_Agent):
    """Synchronous version of A2C algorithm from deepmind paper https://arxiv.org/pdf/1602.01783.pdf"""
    agent_name = "A2C"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.hyperparameters = config.hyperparameters

        # We have an actor and critic in 1 network which outputs logits for each action and a value estimate
        self.actor_critic = self.create_NN(input_dim=self.state_size, output_dim=[self.action_size, 1])
        self.actor_critic_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.hyperparameters["learning_rate"])

        self.episode_number = 0
        self.many_episode_states = []
        self.many_episode_actions = []
        self.many_episode_rewards = []


    def step(self):
        """Runs a step for the A2C agent"""
        self.exploration_epsilon = self.get_updated_epsilon_exploration()
        processors = self.hyperparameters["episodes_per_learning_round"]
        results = self.run_games_in_parallel(processors)

        gradients = [result[0] for result in results]
        self.many_episode_rewards = [result[1] for result in results]

        self.take_optimisation_step(self.actor_critic_optimizer, self.actor_critic, loss=None,
                                    clipping_norm=self.hyperparameters["gradient_clipping_norm"], gradients_given=gradients)

        self.episode_number += self.hyperparameters["episodes_per_learning_round"]
        # self.update_learning_rate(self.hyperparameters["learning_rate"], self.actor_critic_optimizer)

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm, gradients_given=None):
        optimizer.zero_grad() #reset gradients to 0

        if gradients_given is None:
            loss.backward() #this calculates the gradients
        else:
            for ix, parameters in enumerate(network.parameters()):
                for episode in range(len(gradients_given)):
                    if episode == 0:
                        parameters.grad = gradients_given[episode][ix]
                    else:
                        parameters.grad += gradients_given[episode][ix]

        torch.nn.utils.clip_grad_norm_(network.parameters(), clipping_norm) #clip gradients to help stabilise training
        optimizer.step() #this applies the gradients

    def run_games_in_parallel(self, processors):
        """Runs games in parallel. The number of games equals the number of processors use which is given as input"""
        if self.config.use_GPU:
            with closing(GPU_POOL(processes=processors)) as pool:
                results = pool.map(self, range(processors))
                pool.terminate()
        else:
            with closing(Pool(processes=processors)) as pool:
                results = pool.map(self, range(processors))
                pool.terminate()
        return results

    def __call__(self, n):
        exploration = max(0.0, random.uniform(self.exploration_epsilon / 3.0, self.exploration_epsilon * 3.0))
        return self.play_1_episode(exploration)

    def play_1_episode(self, epsilon_exploration):
        """Plays 1 episode using the fixed policy and returns the data"""
        state = self.reset_game_for_worker()
        done = False
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_log_action_probabilities = []
        while not done:
            action, action_log_prob = self.pick_action(self.actor_critic, state, epsilon_exploration)
            self.environment.conduct_action(action)
            next_state = self.environment.get_next_state()
            reward = self.environment.get_reward()
            done = self.environment.get_done()
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_log_action_probabilities.append(action_log_prob)
            state = next_state

        total_loss = self.calculate_total_loss(episode_states, episode_rewards, episode_log_action_probabilities)
        gradients = self.calculate_gradients(total_loss)
        return gradients, episode_rewards, episode_states, episode_actions

    def reset_game_for_worker(self):
        """Resets the game environment so it is ready to play a new episode"""
        seed = random.randint(0, sys.maxsize)
        torch.manual_seed(seed)  # Need to do this otherwise each worker generates same experience
        state = self.environment.reset_environment()
        if self.action_types == "CONTINUOUS": self.noise.reset()
        return state

    def pick_action(self, policy, state, epsilon_exploration=None):
        """Picks an action using the policy"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        actor_output = policy.forward(state)
        actor_output = actor_output[:, list(range(self.action_size))] #we only use first set of columns to decide action, last column is state-value
        # print("Actor outputs should sum to 1 ", actor_output)
        action_distribution = create_actor_distribution(self.action_types, actor_output, self.action_size)
        action = action_distribution.sample().cpu().numpy()
        if self.action_types == "CONTINUOUS": action += self.noise.sample()
        if self.action_types == "DISCRETE":
            if random.random() <= epsilon_exploration:
                action = random.randint(0, self.action_size - 1)
                action = np.array([action])
        action_log_prob = self.calculate_log_action_probability(action, action_distribution)
        return action, action_log_prob

    def calculate_log_action_probability(self, actions, action_distribution):
        """Calculates the log probability of the chosen action"""
        policy_distribution_log_prob = action_distribution.log_prob(torch.Tensor(actions))
        return policy_distribution_log_prob

    def calculate_total_loss(self, episode_states, episode_rewards, episode_log_action_probabilities):
        """Calculates the actor loss + critic loss"""
        discounted_returns = self.calculate_discounted_returns(episode_states, episode_rewards)

        if self.hyperparameters["normalise_rewards"]:
            discounted_returns = self.normalise_discounted_returns(discounted_returns)

        critic_loss, advantages = self.calculate_critic_loss_and_advantages(episode_states, discounted_returns)
        actor_loss = self.calculate_actor_loss(episode_log_action_probabilities, advantages)
        # print("Actor loss ", actor_loss)
        # print("Critic loss ", critic_loss)
        total_loss = actor_loss + critic_loss
        return total_loss

    def calculate_discounted_returns(self, states, rewards):
        """Calculates the cumulative discounted return for an episode which we will then use in a learning iteration"""
        discounted_returns = [0]
        for ix in range(len(states)):
            return_value = rewards[-(ix + 1)] + self.hyperparameters["discount_rate"]*discounted_returns[-1]
            discounted_returns.append(return_value)
        discounted_returns = discounted_returns[1:]
        discounted_returns = discounted_returns[::-1]
        return discounted_returns

    def normalise_discounted_returns(self, discounted_returns):
        """Normalises the discounted returns by dividing by mean and std of returns that episode"""
        mean = np.mean(discounted_returns)
        std = np.std(discounted_returns)
        discounted_returns -= mean
        discounted_returns /= std
        return discounted_returns

    def calculate_critic_loss_and_advantages(self, states, all_discounted_returns):
        """Calculates the critic's loss and the advantages"""
        states = torch.Tensor(states)
        critic_values = self.actor_critic(states)[:, -1]

        # print("Discounted returns ", all_discounted_returns)
        # print("Critic values ", critic_values)

        advantages = torch.Tensor(all_discounted_returns) - critic_values
        advantages = advantages.detach()

        critic_loss =  (torch.Tensor(all_discounted_returns) - critic_values)**2
        critic_loss = critic_loss.mean()

        # print("Advantages ", advantages)

        return critic_loss, advantages

    def calculate_actor_loss(self, action_log_probabilities_for_all_episodes, advantages):
        """Calculates the loss for the actor"""
        action_log_probabilities_for_all_episodes = torch.cat(action_log_probabilities_for_all_episodes)
        actor_loss = -1.0 * action_log_probabilities_for_all_episodes * advantages
        actor_loss = actor_loss.mean()
        return actor_loss

    def calculate_gradients(self, total_loss):
        """Calculates gradients for the worker"""
        self.actor_critic_optimizer.zero_grad()
        total_loss.backward()
        gradients = [param.grad for param in list(self.actor_critic.parameters())]
        return gradients

    def save_result(self):
        """Save the results seen by the agent in the most recent experiences"""
        for ep in range(len(self.many_episode_rewards)):
            total_reward = np.sum(self.many_episode_rewards[ep])
            self.game_full_episode_scores.append(total_reward)
            self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
        self.save_max_result_seen()