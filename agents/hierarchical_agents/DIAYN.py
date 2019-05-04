from gym import Wrapper
from torch import optim, nn
import numpy as np
import random
import copy
from Base_Agent import Base_Agent
from DDQN import DDQN
from SAC import SAC

# NOTE: DIAYN calculates diversity of states penalty each timestep but it might be better to only base it on where the
# agent got to in the last timestep, or after X timesteps

class DIAYN(Base_Agent):
    """Agent based on the paper Diversity is all you need (2018) - https://arxiv.org/pdf/1802.06070.pdf"""
    agent_name = "DIAYN"
    def __init__(self, config):
        super().__init__(config)
        self.training_mode = True
        self.num_skills = config.hyperparameters["num_skills"]
        self.unsupervised_episodes = config.hyperparameters["num_unsupservised_episodes"]
        self.supervised_episodes = config.num_episodes_to_run - self.unsupervised_episodes


        self.discriminator = self.create_NN(1, self.num_skills, key_to_use="DISCRIMINATOR")
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(),
                                              lr=self.hyperparameters["DISCRIMINATOR"]["learning_rate"])
        self.agent_config = copy.deepcopy(config)
        self.agent_config.environment = DIAYN_Skill_Wrapper(self.agent_config.environment)
        self.agent_config.hyperparameters = self.agent_config.hyperparameters["AGENT"]
        self.agent_config.hyperparameters["do_evaluation_iterations"] = False
        self.agent = SAC(self.agent_config)  #We have to use SAC because it involves maximising the policy's entropy over actions which is also a part of DIAYN


        self.timesteps_to_give_up_control_for = self.hyperparameters["MANAGER"]["timesteps_to_give_up_control_for"]
        self.manager_agent_config = copy.deepcopy(config)
        self.manager_agent_config.environment = DIAYN_Manager_Agent_Wrapper(self.agent_config.environment, self.agent, self.timesteps_to_give_up_control_for)
        self.manager_agent_config.hyperparameters = self.manager_agent_config.hyperparameters["MANAGER"]
        self.manager_agent = DDQN(self.manager_agent_config)

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        self.agent.run_n_episodes(num_episodes=self.unsupervised_episodes, show_whether_achieved_goal=False)
        self.manager_agent.run_n_episodes(num_episodes=self.supervised_episodes)

    def disciminator_learn(self, skill, discriminator_outputs):
        if not self.training_mode: return
        assert isinstance(skill, int)
        assert discriminator_outputs.shape[0] == self.num_skills
        assert discriminator_outputs.shape[1] == 5 # should fail..
        loss = nn.CrossEntropyLoss()(discriminator_outputs, skill)
        self.take_optimisation_step(self.discriminator_optimizer, self.discriminator, loss,
                                    self.hyperparameters["DISCRIMINATOR"]["gradient_clipping_norm"])

class DIAYN_Skill_Wrapper(Wrapper):
    """Open AI gym wrapper to help create a pretraining environment in which to train diverse skills according to the
    specification in the Diversity is all you need (2018) paper """
    def __init__(self, env, num_skills, meta_agent):
        Wrapper.__init__(self, env)
        self.num_skills = num_skills
        self.meta_agent = meta_agent
        self.prior_probability_of_skill = 1.0 / self.num_skills #Each skill equally likely to be chosen

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.skill = random.randint(0, self.num_skills - 1)
        return self.observation(observation)

    def observation(self, observation):
        return np.concatenate((np.array(observation), np.array([self.skill])))

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        new_reward, disciminator_outputs = self.calculate_new_reward(next_state)
        self.meta_agent.disciminator_learn(self.skill, disciminator_outputs)
        return self.observation(next_state), new_reward, done, _

    def calculate_new_reward(self, next_state):
        """Calculates an intrinsic reward that encourages maximum exploration. It also keeps track of the discriminator
        outputs so they can be used for training"""
        probability_correct_skill, disciminator_outputs =  self.meta_agent.get_predicted_probability_of_skill(self.skill, next_state)
        new_reward = np.log(probability_correct_skill) - np.log(self.prior_probability_of_skill)
        return new_reward, disciminator_outputs


class DIAYN_Manager_Agent_Wrapper(Wrapper):
    """Environment wrapper for the meta agent. The meta agent uses this environment to take in the state, decide on a skill
     and then grant over control to the lower-level skill for a set number of timesteps"""
    def __init__(self, env, lower_level_agent, timesteps_to_give_up_control_for):
        Wrapper.__init__(self, env)
        self.lower_level_agent = lower_level_agent
        self.timesteps_to_give_up_control_for = timesteps_to_give_up_control_for

    def reset(self, **kwargs):
        self.state = self.env.reset(**kwargs)
        return self.state

    def step(self, skill_chosen):
        """Runs a step in the game from the perspective of the manager agent. This involves giving up control to the
        lower-level agent for a set number of steps"""
        cumulative_reward = 0
        for _ in range(self.timesteps_to_give_up_control_for):
            combined_state = np.concatenate((np.array(self.state), np.array([skill_chosen])))
            action = self.lower_level_agent.pick_action(combined_state, eval_ep=True)
            next_state, reward, done, _ = self.env.step(action)
            cumulative_reward += reward
            self.state = next_state
            if done: break
        return next_state, cumulative_reward, done, _
