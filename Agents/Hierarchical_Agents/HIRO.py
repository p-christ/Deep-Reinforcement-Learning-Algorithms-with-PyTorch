from gym import Wrapper

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

        self.high_level_policy =

class Goal_Wrapper(Wrapper):
    """Open AI gym wrapper to help create an environment where a goal from a higher-level agent is treated as part
    of the environment state"""
    def __init__(self, env, HIRO_agent, max_timesteps):
        Wrapper.__init__(self, env)
        self.env = env
        self.HIRO_agent = HIRO_agent
        self.max_timesteps = max_timesteps

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.goal =

        return self.observation(observation)

    def observation(self, observation):
        return np.concatenate((np.array(observation), np.array([self.skill])))

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        self.update_state_visitations(next_state)
        probability_correct_skill = self.calculate_probability_correct_skill(next_state)
        new_reward = reward + self.regularisation_weight * np.log(probability_correct_skill)
        return self.observation(next_state), new_reward, done, _







