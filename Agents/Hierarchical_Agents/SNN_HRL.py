import copy
import random
import numpy as np
from Agents.Base_Agent import Base_Agent
from Agents.Policy_Gradient_Agents.PPO import PPO


class SNN_HRL(Base_Agent):
    """Implements the hierarchical RL agent that uses stochastic neural networks (SNN) from the paper Florensa et al. 2017
    https://arxiv.org/pdf/1704.03012.pdf

    Works by:
    1) Creating a pre-training environment within which the skill_agent can learn for some period of time
    2) Then skill_agent is frozen
    3) Then we train a manager agent that chooses which of the pre-trained skills to let act for it for some period of time

    Note that it only works with discrete states at the moment.
    """
    agent_name = "SNN-HRL"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert isinstance(self.environment.state, int), "only works for discrete states currently"
        self.env_parameters = config.env_parameters
        self.num_skills = self.hyperparameters["SKILL_AGENT"]["num_skills"]

        self.skill_agent_config = copy.deepcopy(config)
        self.skill_agent_config.hyperparameters = self.skill_agent_config.hyperparameters["SKILL_AGENT"]
        self.skill_agent_config.state_size = self.state_size + 1

        self.skill = 1
        self.create_skill_learning_environment()
        self.skill_agent_config.environment = self.skills_env

        self.skill_agent = PPO(self.skill_agent_config)
        self.episodes_for_pretraining = 100

        self.pretraining_skills()

    def pretraining_skills(self):
        """Runs pretraining during which time skills are learnt by the skills agent"""
        self.skill_agent_config.num_episodes_to_run = self.episodes_for_pretraining
        self.skills_agent.run_n_episodes()

    def create_skill_learning_environment(self):
        """Creates the environment that the skills agent will use to learn skills during pretraining"""

        meta_agent = self
        environment_class = self.environment.__class__

        class skills_env(environment_class):
            """Creates an environment from within which to train skills"""
            def __init__(self, meta_agent):
                environment_class.__init__(self, **meta_agent.env_parameters)
                self.meta_agent = meta_agent
                self.state_visitations = [[0 for _ in range(meta_agent.environment.observation_space.n)] for _ in
                                          range(self.meta_agent.num_skills)]
                self.regularisation_weight = self.meta_agent.skill_agent_config.hyperparameters["regularisation_weight"]
                self.visitations_decay = self.meta_agent.skill_agent_config.hyperparameters["visitations_decay"]

            def reset(self):
                environment_class.reset(self)
                return np.array([self.state, meta_agent.skill])

            def step(self, action):
                next_state, reward, done, _ = environment_class.step(self, action)
                self.update_state_visitations(next_state)
                probability_correct_skill = self.calculate_probability_correct_skill(next_state)
                new_reward = reward + self.regularisation_weight * np.log(probability_correct_skill)
                return np.array([next_state, meta_agent.skill]), new_reward, done, _

            def calculate_probability_correct_skill(self, next_state):
                """Calculates the probability that a certain"""
                visitations_correct_skill = self.state_visitations[self.meta_agent.skill][next_state]
                visitations_any_skill = np.sum([visit[next_state] for visit in self.state_visitations])
                probability = float(visitations_correct_skill) / float(visitations_any_skill)
                return probability

            def update_state_visitations(self, next_state):
                self.state_visitations = [[val * self.visitations_decay for val in sublist] for sublist in
                                          self.state_visitations]
                self.state_visitations[self.meta_agent.skill][next_state] += 1

        self.skills_env = skills_env(meta_agent)
