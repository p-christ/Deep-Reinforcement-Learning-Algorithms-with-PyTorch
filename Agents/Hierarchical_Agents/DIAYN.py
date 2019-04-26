import time

from gym import Wrapper
from torch import optim, nn
import numpy as np
import random
import copy

class DIAYN(object):
    """Agent based on the paper Diversity is all you need (2018) - https://arxiv.org/pdf/1802.06070.pdf"""
    agent_name = "DIAYN"
    def __init__(self, config):
        super().__init__(config)
        self.training_mode = True

        self.num_skills = config.hyperparameters["num_skills"]
        self.discriminator = self.create_NN(1, self.num_skills, key_to_use="DISCRIMINATOR")
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(),
                                              lr=self.hyperparameters["DISCRIMINATOR"]["learning_rate"])
        self.discriminator_loss_fn = nn.CrossEntropyLoss()

        self.agent_config = copy.deepcopy(config)
        self.agent_config.hyperparameters = self.agent_config.hyperparameters["AGENT"]
        self.agent = SAC(self.agent_config)  #We have to use SAC because it involves maximising the policy's entropy over actions which is also a part of DIAYN

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):

        self.agent.run_n_episodes()

        # then we evaluate the range of states visited by each skill

        #
        #
        # if num_episodes is None: num_episodes = self.config.  num_episodes_to_run
        # start = time.time()
        # while self.episode_number < num_episodes:
        #     self.reset_game()
        #     self.step()
        #     if save_and_print_results: self.save_and_print_result()
        # time_taken = time.time() - start
        # if show_whether_achieved_goal: self.show_whether_achieved_goal()
        # if self.config.save_model: self.locally_save_policy()
        # return self.game_full_episode_scores, self.rolling_results, time_taken


    def disciminator_learn(self, skill, discriminator_outputs):
        if not self.training_mode: return

        assert isinstance(skill, int)
        assert discriminator_outputs.shape[0] == self.num_skills
        assert discriminator_outputs.shape[1] == 5 # should fail..

        loss = self.discriminator_loss_fn(discriminator_outputs, skill)

        self.take_optimisation_step(self.discriminator_optimizer, self.discriminator, loss,
                                    self.hyperparameters["DISCRIMINATOR"]["gradient_clipping_norm"])



        # calculate loss
        # do optimisation step on it and optimizer




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