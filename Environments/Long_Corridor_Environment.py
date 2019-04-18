import random
import numpy as np
from Environments.Base_Environment import Base_Environment

class Long_Corridor_Environment(Base_Environment):
    """Is the environment from pg.6 of the paper Hierarchical Deep Reinforcement Learning: Integrating Temporal
    Abstraction and Intrinsic Motivation.
    https://papers.nips.cc/paper/6233-hierarchical-deep-reinforcement-learning-integrating-temporal-abstraction-and-intrinsic-motivation.pdf"""
    environment_name = "Long Corridor Environment"

    def __init__(self, num_states=6, stochasticity_of_action_right=0.5):
        self.reset_environment()
        self.next_state = None
        self.reward = None
        self.done = False
        self.action_translation = {0: "left", 1: "right"}
        self.stochasticity_of_action_right = stochasticity_of_action_right
        self.num_states = num_states
        self.visited_final_state = False
        self.reward_if_visited_final_state = 1.0
        self.reward_if_havent_visited_final_state = 0.01

    def reset_environment(self):
        self.state = 1 #environment always starts in state 1
        self.next_state = None
        self.reward = None
        self.done = False
        self.visited_final_state = False
        self.episode_steps = 0
        return np.array([self.state])

    def conduct_action(self, action):
        self.episode_steps += 1
        if type(action) is np.ndarray:
            action = action[0]
        assert action in [0, 1], "Action must be a 0 or a 1"

        if action == 0: self.move_left()
        else: self.move_right()

        self.update_done_reward_and_visited_final_state()
        self.state = self.next_state

    def update_done_reward_and_visited_final_state(self):
        if self.next_state == 0:
            self.done = True
            if self.visited_final_state: self.reward = self.reward_if_visited_final_state
            else: self.reward = self.reward_if_havent_visited_final_state
        else:
            self.reward = 0
        if self.next_state == self.num_states - 1: self.visited_final_state = True
        if self.episode_steps >= self.get_max_steps_per_episode(): self.done = True

    def move_left(self):
        """Moves left in environment"""
        self.next_state = self.state - 1

    def move_right(self):
        """Moves right in environment"""
        if random.random() < self.stochasticity_of_action_right:
            self.next_state = self.state - 1
        else:
            self.next_state = min(self.state + 1, self.num_states - 1)

    def get_state(self):
        return np.array([self.state])

    def get_action_size(self):
        return 2

    def get_state_size(self):
        return 1

    def get_num_possible_states(self):
        return self.num_states

    def get_next_state(self):
        return np.array([self.next_state])

    def get_reward(self):
        return self.reward

    def get_done(self):
        return self.done

    def get_max_steps_per_episode(self):
        return 100

    def get_action_types(self):
        return "DISCRETE"

    def get_score_to_win(self):
        return 1.0

    def get_rolling_period_to_calculate_score_over(self):
        return 100

