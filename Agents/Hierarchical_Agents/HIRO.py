import copy

from gym import Wrapper
import numpy as np
from Base_Agent import Base_Agent
from DDPG import DDPG
from Trainer import Trainer


class HIRO(Base_Agent):
    agent_name = "HIRO"

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

        self.max_sub_policy_timesteps = config.hyperparameters["LOWER_LEVEL"]["max_lower_level_timesteps"]
        self.config.hyperparameters = Trainer.add_default_hyperparameters_if_not_overriden({"OPEN": self.config.hyperparameters})
        self.config.hyperparameters = self.config.hyperparameters["OPEN"]

        print(self.config.hyperparameters)
        self.state = None #true state of environment
        self.next_state = None
        self.reward = None
        self.done = False
        self.goal = None
        self.external_state = None #state of environment with goal appended
        self.external_next_state = None
        self.higher_level_step_reward = 0

        self.lower_level_agent_config = copy.deepcopy(config)
        self.lower_level_agent_config.hyperparameters = self.lower_level_agent_config.hyperparameters["LOWER_LEVEL"]

        # self.lower_level_agent_config.hyperparameters = Trainer.add_default_hyperparameters_if_not_overriden(self.lower_level_agent_config.hyperparameters)

        self.lower_level_agent_config.environment = Lower_Level_Agent_Environment_Wrapper(self.environment, self, self.max_sub_policy_timesteps)
        self.lower_level_agent = DDPG(self.lower_level_agent_config)

        self.lower_level_agent.average_score_required_to_win = float("inf")

        print("LOWER LEVEL actor {} to {}".format(self.lower_level_agent.actor_local.input_dim, self.lower_level_agent.actor_local.output_dim))

        self.higher_level_agent_config = copy.deepcopy(config)
        self.higher_level_agent_config.hyperparameters = self.higher_level_agent_config.hyperparameters["HIGHER_LEVEL"]
        self.higher_level_agent_config.environment = Higher_Level_Agent_Environment_Wrapper(self.environment, self)
        self.higher_level_agent = DDPG(self.higher_level_agent_config)

        print("HIGHER LEVEL actor {} to {}".format(self.higher_level_agent.actor_local.input_dim,
                                                  self.higher_level_agent.actor_local.output_dim))





        self.extrinsic_rewards = []

    def run_n_episodes(self):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        self.higher_level_agent.run_n_episodes(self.config.num_episodes_to_run)
        #
        #
        # sub_policy_agent = DDPG(self.sub_policy_config)
        #
        # # for each iteration of a manager step...  sub policy must do a whole step
        #
        #
        # # high level policy takes in state and produces goal state
        # # goal state set for lower-level policy which then acts until done...
        # # if episode not done then high level policy takes in state and produces goal state
        #

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def get_external_state(self):
        return self.external_state

    def set_external_state(self, external_state):
        self.external_state = external_state

    def track_higher_level_step_reward(self, reward):
        self.higher_level_step_reward += reward

    def get_higher_level_step_reward(self):
        return self.higher_level_step_reward

    def set_goal_for_lower_level(self, goal):
        self.goal = goal

    def set_done(self, done):
        """Sets boolean for whether the episode is over for high-level agent"""
        self.done = done

    def get_done(self):
        return self.done



    # def give_next_goal_for_sub_policy(self):
    #     """Provides the next goal for the sub policy to try and achieve"""
    #
    #     print("must update this")
    #
    #     return np.array([0.0, 0.0, 0.0])

    # def save_extrinsic_rewards(self, reward):
    #     self.extrinsic_rewards.append(reward)


class Higher_Level_Agent_Environment_Wrapper(Wrapper):

    def __init__(self, env, HIRO_agent):
        Wrapper.__init__(self, env)
        self.env = env
        self.HIRO_agent = HIRO_agent
        self.action_space = self.observation_space


    def reset(self, **kwargs):
        print("Higher level resetting the game")
        self.HIRO_agent.set_state(self.env.reset(**kwargs))
        self.lower_level_timesteps = 0
        self.HIRO_agent.done = False
        return self.HIRO_agent.state

    def step(self, goal):
        # print("Step for higher agent")
        self.HIRO_agent.higher_level_step_reward = 0

        self.HIRO_agent.set_goal_for_lower_level(goal)
        print("RUNNING LOW LEVEL EPISODE")
        self.HIRO_agent.lower_level_agent.episode_number = 0
        self.HIRO_agent.lower_level_agent.run_n_episodes(num_episodes=1, show_whether_achieved_goal=False)

        last_state = self.HIRO_agent.get_state()
        reward = self.HIRO_agent.get_higher_level_step_reward()
        done = self.HIRO_agent.get_done()

        print("Higher level done ", done)

        # print("Higher agent done ", done)

        return last_state, reward, done, {}



class Lower_Level_Agent_Environment_Wrapper(Wrapper):
    """Open AI gym wrapper to help create an environment where a goal from a higher-level agent is treated as part
    of the environment state"""
    def __init__(self, env, HIRO_agent, max_sub_policy_timesteps):
        Wrapper.__init__(self, env)
        self.env = env
        self.HIRO_agent = HIRO_agent
        self.max_sub_policy_timesteps = max_sub_policy_timesteps

    def reset(self, **kwargs):
        print("Lower level resetting the game")

        # Must offer option when values are None so that state_size can be calculated when initialising the base agent

        print(self.HIRO_agent)

        if self.HIRO_agent.external_state is not None: state = self.HIRO_agent.external_state
        else: state = self.env.reset()

        if self.HIRO_agent.goal is not None: goal = self.HIRO_agent.goal
        else: goal = state

        self.lower_level_timesteps = 0
        self.lower_level_turn_over = False

        # print("High level state ", state)
        # print("High level goal ", goal)
        #
        # print("LOW LEVEL STATE ", self.turn_internal_state_to_external_state(state, goal))

        return self.turn_internal_state_to_external_state(state, goal)

    def turn_internal_state_to_external_state(self, internal_state, goal):
        return np.concatenate((np.array(internal_state), goal))

    def step(self, action):
        print("Stepping in low level")
        self.lower_level_timesteps += 1
        self.HIRO_agent.next_state, reward, done, _ = self.env.step(action)

        # need to think about what else to save in main agent
        self.HIRO_agent.track_higher_level_step_reward(reward)
        #
        # self.HIRO_agent.save_extrinsic_rewards(reward)

        intrinsic_reward = self.calculate_intrinsic_reward(self.HIRO_agent.state, self.HIRO_agent.next_state, self.HIRO_agent.goal)

        self.HIRO_agent.state = self.HIRO_agent.next_state

        self.HIRO_agent.set_done(done)

        self.lower_level_turn_over = done or self.lower_level_timesteps >= self.max_sub_policy_timesteps

        print("Lower level done ", done)
        print("Lower level turn over ", self.lower_level_turn_over)

        # print("Lower level over ", self.lower_level_episode_over)

        return self.turn_internal_state_to_external_state(self.HIRO_agent.next_state, self.HIRO_agent.goal), intrinsic_reward, self.lower_level_turn_over, _

    def calculate_intrinsic_reward(self, internal_state, internal_next_state, goal):
        """Calculates the intrinsic reward for the agent according to whether it has made progress towards the goal
        or not since the last timestep"""
        desired_next_state = internal_state + goal
        error = desired_next_state - internal_next_state
        return -(np.dot(error, error))**0.5






