import copy
import torch
from contextlib import closing
from torch.distributions import Categorical
from multiprocessing import Pool
from random import randint



class Parallel_Experience_Generator(object):
    """ Plays n episode in parallel using a fixed agent"""

    def __init__(self, environment, model):

        self.environment =  environment
        self.model = model

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
        env, policy, state = self.reset_game()
        done = False

        episode_states = []
        episode_actions = []
        episode_rewards = []

        while not done:
            action = self.pick_action(policy, state)
            env.conduct_action(action)
            next_state = env.get_next_state()
            reward = env.get_reward()
            done = env.get_done()

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state

        return episode_states, episode_actions, episode_rewards

    def reset_game(self):
        """Resets the game environment so it is ready to play a new episode"""
        seed = randint(10, 500)
        torch.manual_seed(seed) # Need to do this otherwise each worker generates same experience
        env = copy.deepcopy(self.environment)
        policy = copy.deepcopy(self.model)
        env.reset_environment()
        state = env.get_state()
        return env, policy, state

    def pick_action(self, policy, state):
        """Picks an action using the policy"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        new_policy_action_probabilities = policy.forward(state).cpu()
        action_distribution = Categorical(new_policy_action_probabilities)  # this creates a distribution to sample from
        action = action_distribution.sample().numpy()[0]
        return action


