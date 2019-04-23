import copy

from gym import Wrapper
import numpy as np
from Base_Agent import Base_Agent


class HIRO(Base_Agent):

    """

    high-level policy sets goals directly equivalent to states for lower level every c steps
    can use a goal transition function which changes the goal as we go through the c steps but you can also just leave it...
    they use a goal transition function (see paper )

    lower-level observes state and goal and produces low level action. receives intrinsic rewards

    both low level and high level can be learnt off-policy. we use DDPG for both
    off-policy correction is used for higher-level policy



    """


    def __init__(self, config):
        Base_Agent.__init__(self, config)

        self.max_sub_policy_timesteps = config.hyperparameters["max_sub_policy_timesteps"]

        self.env_for_sub_policy = Goal_Wrapper(copy.deepcopy(self.environment), self, self.max_sub_policy_timesteps)


        self.extrinsic_rewards = []

    def give_latest_goal(self):

        print("must update this")

        return np.array([0.0, 0.0, 0.0])

    def save_extrinsic_rewards(self, reward):
        self.extrinsic_rewards.append(reward)


class Goal_Wrapper(Wrapper):
    """Open AI gym wrapper to help create an environment where a goal from a higher-level agent is treated as part
    of the environment state"""
    def __init__(self, env, HIRO_agent, max_sub_policy_timesteps):
        Wrapper.__init__(self, env)
        self.env = env
        self.HIRO_agent = HIRO_agent
        self.max_sub_policy_timesteps = max_sub_policy_timesteps

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.internal_state = observation
        self.goal = self.HIRO_agent.give_latest_goal()
        self.episode_over = False
        self.timesteps = 0
        return self.turn_internal_state_to_external_state(observation)

    def turn_internal_state_to_external_state(self, internal_state):
        return np.concatenate((np.array(internal_state), self.goal))

    def step(self, action):
        self.timesteps += 1
        self.internal_next_state, reward, done, _ = self.env.step(action)

        # need to think about what else to save in main agent
        self.HIRO_agent.save_extrinsic_rewards(reward)

        intrinsic_reward = self.calculate_intrinsic_reward(self.internal_state, self.internal_next_state, self.goal)

        self.internal_state = self.internal_next_state

        self.episode_over = done
        self.sub_policy_episode_over = done or self.timesteps >= self.max_sub_policy_timesteps

        return self.turn_internal_state_to_external_state(self.internal_next_state), intrinsic_reward, self.sub_policy_episode_over, _

    def calculate_intrinsic_reward(self, internal_state, internal_next_state, goal):
        """Calculates the intrinsic reward for the agent according to whether it has made progress towards the goal
        or not since the last timestep"""

        # print(internal_state)
        # print(internal_next_state)
        # print(goal)
        combined = internal_state + goal - internal_next_state
        # print(combined)
        # print(combined**2)
        # print((combined**2)**0.5)
        #
        # print(-(np.dot(combined, combined))**0.5)

        return -(np.dot(combined, combined))**0.5






