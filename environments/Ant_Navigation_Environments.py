from .ant_environments.create_maze_env import create_maze_env
import numpy as np

"""Environments taken from HIRO paper github repo: https://github.com/tensorflow/models/tree/master/research/efficient-hrl
There are three environments that can be represented by this class depending on what environment_name you provide. 
The options are: ["AntMaze", "AntPush", "AntFall"].

Note that "Success" for this game is defined by the authors as achieving -5 or more on the last step of the episode 
but that this isn't coded in anyway as part of the environment. 
"""
class Ant_Navigation_Environments(object):

    def __init__(self, environment_name):
        self.environment_name = environment_name
        self.base_env = create_maze_env(env_name=self.environment_name).gym  #

        self.goal_sample_fn = self.get_goal_fn()
        self.reward_fn = self.get_reward_fn()
        self.goal = None

        self.unwrapped = self.base_env.unwrapped
        self.spec = self.base_env.spec
        self.action_space = self.base_env.action_space

    def reset(self):
        self.steps_taken = 0
        obs = self.base_env.reset()
        self.goal = self.goal_sample_fn()
        return np.concatenate([obs, self.goal])

    def step(self, action):
        self.steps_taken += 1
        obs, _, _, info = self.base_env.step(action)
        reward = self.reward_fn(obs, self.goal)
        done = self.steps_taken >= 500
        return np.concatenate([obs, self.goal]), reward, done, info

    def get_goal_fn(self):
        """Produces the function required to generate a goal for each environment"""
        if self.environment_name == "AntMaze":
            return lambda: np.array([0., 16.])
            #Can also use np.random.uniform((-4, -4), (20, 20)) for training purposes
        elif self.environment_name == "AntPush":
            return lambda: np.array([0., 19.])
        elif self.environment_name == "AntFall":
            return lambda: np.array([0., 27., 4.5])
        else:
            raise ValueError("Unknown environment name")

    def get_reward_fn(self):
        """Provides function required to calculates rewards for each game"""
        if self.environment_name == 'AntMaze':
            return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
        elif self.environment_name == 'AntPush':
            return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
        elif self.environment_name == 'AntFall':
            return lambda obs, goal: -np.sum(np.square(obs[:3] - goal)) ** 0.5
        else:
            raise ValueError("Unknown environment name")




