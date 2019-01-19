import gym
import numpy as np
from Environments.Base_Environment import Base_Environment

class Fetch_Reach_Environment(Base_Environment):

    def __init__(self):
        self.game_environment = gym.make("FetchReach-v1")
        self.state_information = self.game_environment.reset()
        self.desired_goal = self.state_information["desired_goal"]
        self.achieved_goal = self.state_information["achieved_goal"]
        self.state = np.concatenate((self.state_information["observation"], self.desired_goal), axis=None)
        self.next_state = None
        self.reward = None
        self.done = False
        self.info = None
        self.reward_for_achieving_goal = 0
        self.step_reward_for_not_achieving_goal = -1

    def conduct_action(self, action):
        self.state_information, self.reward, self.done, self.info = self.game_environment.step(action)
        self.desired_goal = self.state_information["desired_goal"]
        self.achieved_goal = self.state_information["achieved_goal"]
        self.next_state = np.concatenate((self.state_information["observation"], self.desired_goal), axis=None)
        self.state = self.next_state

    def get_action_size(self):
        return 4

    def get_state_size(self):
        return len(self.state)

    def get_state(self):
        return self.state

    def get_next_state(self):
        return self.next_state

    def get_reward(self):
        return self.reward

    def get_done(self):
        return self.done

    def get_current_reward_for_another_goal(self, goal):
        return self.game_environment.compute_reward(self.achieved_goal, goal, self.info)

    def get_desired_goal(self):
        return self.desired_goal

    def get_achieved_goal(self):
        return self.achieved_goal

    def reset_environment(self):
        self.state_information = self.game_environment.reset()
        self.desired_goal = self.state_information["desired_goal"]
        self.achieved_goal = self.state_information["achieved_goal"]
        self.state = np.concatenate((self.state_information["observation"], self.desired_goal), axis=None)

    def get_reward_for_achieving_goal(self):
        return self.reward_for_achieving_goal

    def get_step_reward_for_not_achieving_goal(self):
        return self.step_reward_for_not_achieving_goal

    def get_max_steps_per_episode(self):
        return 50

    def get_action_types(self):
        return "CONTINUOUS"

    def get_score_to_win(self):
        return -5

    def get_rolling_period_to_calculate_score_over(self):
        return 100

