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

        self.higher_level_state = None #true state of environment
        self.higher_level_next_state = None

        self.higher_level_reward = None
        self.lower_level_reward = None

        self.higher_level_done = False
        self.lower_level_done = False

        self.goal = None

        self.lower_level_state = None #state of environment with goal appended
        self.lower_level_next_state = None

        self.lower_level_agent_config = copy.deepcopy(config)
        self.lower_level_agent_config.hyperparameters = self.lower_level_agent_config.hyperparameters["LOWER_LEVEL"]

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


    def run_n_episodes(self):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        self.higher_level_agent.run_n_episodes(self.config.num_episodes_to_run)

    def goal_transition(self, next_state):
        """Updates the goal """
        return self.higher_level_state + self.goal - next_state



class Higher_Level_Agent_Environment_Wrapper(Wrapper):

    def __init__(self, env, HIRO_agent):
        Wrapper.__init__(self, env)
        self.env = env
        self.HIRO_agent = HIRO_agent
        self.action_space = self.observation_space


    def reset(self, **kwargs):
        self.HIRO_agent.higher_level_state = self.env.reset(**kwargs)
        return self.HIRO_agent.higher_level_state

    def step(self, goal):
        self.HIRO_agent.higher_level_reward = 0
        self.HIRO_agent.goal = goal

        self.HIRO_agent.lower_level_agent.episode_number = 0 #must reset lower level agent to 0 episodes completed otherwise won't run more episodes
        self.HIRO_agent.lower_level_agent.run_n_episodes(num_episodes=1, show_whether_achieved_goal=False)
        return self.HIRO_agent.higher_level_next_state, self.HIRO_agent.higher_level_reward, self.HIRO_agent.higher_level_done, {}




class Lower_Level_Agent_Environment_Wrapper(Wrapper):
    """Open AI gym wrapper to help create an environment where a goal from a higher-level agent is treated as part
    of the environment state"""
    def __init__(self, env, HIRO_agent, max_sub_policy_timesteps):
        Wrapper.__init__(self, env)
        self.env = env
        self.HIRO_agent = HIRO_agent
        self.max_sub_policy_timesteps = max_sub_policy_timesteps

    def reset(self, **kwargs):
        if self.HIRO_agent.higher_level_state is not None: state = self.HIRO_agent.higher_level_state
        else:
            print("INITIATION ONLY")
            state = self.env.reset()

        if self.HIRO_agent.goal is not None: goal = self.HIRO_agent.goal
        else:
            print("INITIATION ONLY")
            goal = state

        self.lower_level_timesteps = 0
        self.HIRO_agent.lower_level_done = False
        return self.turn_internal_state_to_external_state(state, goal)

    def turn_internal_state_to_external_state(self, internal_state, goal):
        print("Internal state ", internal_state)
        print("Goal ", goal )
        return np.concatenate((np.array(internal_state), goal))

    def step(self, action):

        self.lower_level_timesteps += 1
        next_state, extrinsic_reward, done, _ = self.env.step(action)

        print("extrinsic reward ", extrinsic_reward)

        self.update_rewards(extrinsic_reward, next_state)
        self.update_goal(next_state)
        self.update_state_and_next_state(next_state)
        self.update_done(done)

        return self.HIRO_agent.lower_level_next_state, self.HIRO_agent.lower_level_reward, self.HIRO_agent.lower_level_done, _

    def update_rewards(self, extrinsic_reward, next_state):
        self.HIRO_agent.higher_level_reward += extrinsic_reward
        self.HIRO_agent.lower_level_reward = self.calculate_intrinsic_reward(self.HIRO_agent.higher_level_state,
                                                                             next_state,
                                                                             self.HIRO_agent.goal)
    def update_goal(self, next_state):
        self.HIRO_agent.goal = self.HIRO_agent.goal_transition(next_state)

    def update_state_and_next_state(self, next_state):
        self.HIRO_agent.higher_level_next_state = next_state
        self.HIRO_agent.lower_level_next_state = self.turn_internal_state_to_external_state(next_state,
                                                                                            self.HIRO_agent.goal)
        self.HIRO_agent.higher_level_state = self.HIRO_agent.higher_level_next_state
        self.HIRO_agent.lower_level_state = self.HIRO_agent.lower_level_next_state

    def update_done(self, done):
        self.HIRO_agent.higher_level_done = done
        self.HIRO_agent.lower_level_done = done or self.lower_level_timesteps >= self.max_sub_policy_timesteps


    def calculate_intrinsic_reward(self, internal_state, internal_next_state, goal):
        """Calculates the intrinsic reward for the agent according to whether it has made progress towards the goal
        or not since the last timestep"""
        print("internal state ", internal_state)
        print("internal next state ", internal_next_state)
        print("goal ", goal)

        desired_next_state = internal_state + goal
        error = desired_next_state - internal_next_state
        intrinsic_reward = -(np.dot(error, error))**0.5

        print("intrinsic reward ", intrinsic_reward)
        return intrinsic_reward






