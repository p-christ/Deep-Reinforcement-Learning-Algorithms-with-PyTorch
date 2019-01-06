import torch
from contextlib import closing
from torch.distributions import Categorical
from multiprocessing import Pool
from random import randint


class Parallel_Experience_Generator(object):
    """ Plays n episode in parallel using a fixed agent"""

    def __init__(self, environment, policy):

        self.environment =  environment
        self.policy = policy

    def play_n_episodes(self, n):
        """Plays n episodes in parallel using the fixed policy and returns the data"""

        with closing(Pool(processes=n)) as pool:
            results = pool.map(self, range(n))
            pool.terminate()

        states_for_all_episodes = [episode[0] for episode in results]
        actions_for_all_episodes = [episode[1] for episode in results]
        rewards_for_all_episodes = [episode[2] for episode in results]

        return states_for_all_episodes, actions_for_all_episodes, rewards_for_all_episodes

    def __call__(self, n):
        return self.play_1_episode()

    def play_1_episode(self):
        """Plays 1 episode using the fixed policy and returns the data"""
        state = self.reset_game()
        done = False

        episode_states = []
        episode_actions = []
        episode_rewards = []

        while not done:
            action = self.pick_action(self.policy, state)
            self.environment.conduct_action(action)
            next_state = self.environment.get_next_state()
            reward = self.environment.get_reward()
            done = self.environment.get_done()

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state

        return episode_states, episode_actions, episode_rewards

    def reset_game(self):
        """Resets the game environment so it is ready to play a new episode"""
        seed = randint(0, 10000)
        torch.manual_seed(seed) # Need to do this otherwise each worker generates same experience
        self.environment.reset_environment()
        state = self.environment.get_state()
        return state

    def pick_action(self, policy, state):
        """Picks an action using the policy"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        new_policy_action_probabilities = policy.forward(state)
        action_distribution = Categorical(new_policy_action_probabilities)  # this creates a distribution to sample from
        action = action_distribution.sample().numpy()[0]
        return action


