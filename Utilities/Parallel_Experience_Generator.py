import torch
from contextlib import closing
from multiprocessing import Pool
from torch.multiprocessing import Pool as GPU_POOL
from random import randint
from Utilities.OU_Noise import OU_Noise
from Utilities.Utility_Functions import create_actor_distribution

class Parallel_Experience_Generator(object):
    """ Plays n episode in parallel using a fixed agent. Only works for PPO or DDPG type agents at the moment, not Q-learning agents"""
    def __init__(self, environment, policy, seed, hyperparameters, use_GPU=False):
        self.use_GPU = use_GPU
        self.environment =  environment
        self.action_size = self.environment.get_action_size()
        self.action_types = self.environment.get_action_types()
        self.policy = policy
        self.hyperparameters = hyperparameters
        self.noise = OU_Noise(self.action_size, seed, self.hyperparameters["mu"],
                            self.hyperparameters["theta"], self.hyperparameters["sigma"])

    def play_n_episodes(self, n):
        """Plays n episodes in parallel using the fixed policy and returns the data"""
        if self.use_GPU:
            with closing(GPU_POOL(processes=n)) as pool:
                results = pool.map(self, range(n))
                pool.terminate()
        else:
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
        self.noise.reset()
        state = self.environment.get_state()
        return state

    def pick_action(self, policy, state):
        """Picks an action using the policy"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        actor_output = policy.forward(state)
        action_distribution = create_actor_distribution(self.action_types, actor_output, self.action_size)
        action = action_distribution.sample().cpu().numpy()
        if self.action_types == "CONTINUOUS":
            action += self.noise.sample()
        return action