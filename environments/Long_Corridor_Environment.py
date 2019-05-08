import random
from collections import namedtuple
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

class Long_Corridor_Environment(gym.Env):
    """Is the environment from pg.6 of the paper Hierarchical Deep Reinforcement Learning: Integrating Temporal
    Abstraction and Intrinsic Motivation.
    https://papers.nips.cc/paper/6233-hierarchical-deep-reinforcement-learning-integrating-temporal-abstraction-and-intrinsic-motivation.pdf"""
    environment_name = "Long Corridor Environment"

    def __init__(self, num_states=6, stochasticity_of_action_right=0.5):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(num_states)
        self.seed()
        self.reward_threshold = 1.0
        self.trials = 100
        self.max_episode_steps = 100
        self.id = "Long Corridor"
        self.action_translation = {0: "left", 1: "right"}
        self.stochasticity_of_action_right = stochasticity_of_action_right
        self.num_states = num_states
        self.visited_final_state = False
        self.reward_if_visited_final_state = 1.0
        self.reward_if_havent_visited_final_state = 0.01

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.episode_steps += 1
        if type(action) is np.ndarray:
            action = action[0]
        assert action in [0, 1], "Action must be a 0 or a 1"
        if action == 0: self.move_left()
        else: self.move_right()
        self.update_done_reward_and_visited_final_state()
        self.state = self.next_state
        self.s = np.array(self.next_state)
        return self.s, self.reward, self.done, {}

    def reset(self):
        self.state = 1 #environment always starts in state 1
        self.next_state = None
        self.reward = None
        self.done = False
        self.visited_final_state = False
        self.episode_steps = 0
        self.s = np.array(self.state)
        return self.s

    def update_done_reward_and_visited_final_state(self):
        if self.next_state == 0:
            self.done = True
            if self.visited_final_state: self.reward = self.reward_if_visited_final_state
            else: self.reward = self.reward_if_havent_visited_final_state
        else:
            self.reward = 0
        if self.next_state == self.num_states - 1: self.visited_final_state = True
        if self.episode_steps >= self.max_episode_steps: self.done = True

    def move_left(self):
        """Moves left in environment"""
        self.next_state = self.state - 1

    def move_right(self):
        """Moves right in environment"""
        if random.random() < self.stochasticity_of_action_right: self.next_state = self.state - 1
        else: self.next_state = min(self.state + 1, self.num_states - 1)
