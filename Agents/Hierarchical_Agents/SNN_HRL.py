import copy
import random
import time
from collections import namedtuple
import numpy as np
from Agents.Base_Agent import Base_Agent
from Agents.Policy_Gradient_Agents.PPO import PPO
from DQN import DQN

class SNN_HRL(Base_Agent):
    """Implements the hierarchical RL agent that uses stochastic neural networks (SNN) from the paper Florensa et al. 2017
    https://arxiv.org/pdf/1704.03012.pdf
    Works by:
    1) Creating a pre-training environment within which the skill_agent can learn for some period of time
    2) Then skill_agent is frozen
    3) Then we train a manager agent that chooses which of the pre-trained skills to let act for it for some period of time
    Note that it only works with discrete states at the moment."""
    agent_name = "SNN-HRL"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert isinstance(self.environment.state, int), "only works for discrete states currently"
        self.env_parameters = config.env_parameters
        self.num_skills = self.hyperparameters["SKILL_AGENT"]["num_skills"]
        self.episodes_for_pretraining =  self.hyperparameters["SKILL_AGENT"]["episodes_for_pretraining"]
        self.timesteps_before_changing_skill = self.hyperparameters["MANAGER"]["timesteps_before_changing_skill"]

        self.skill_agent_config = copy.deepcopy(config)
        self.skill_agent_config.hyperparameters = self.skill_agent_config.hyperparameters["SKILL_AGENT"]
        self.skill_agent_config.num_episodes_to_run = self.episodes_for_pretraining

        self.manager_config = copy.deepcopy(config)
        self.manager_config.hyperparameters = self.manager_config.hyperparameters["MANAGER"]

    def create_skill_learning_environment(self, num_skills, regularisation_weight, visitations_decay, env_parameters,
                                          num_states):
        """Creates the environment that the skills agent will use to learn skills during pretraining"""
        # meta_agent = self
        environment_class = self.environment.__class__

        class skills_env(environment_class):
            """Creates an environment from within which to train skills"""
            def __init__(self):  # , meta_agent):
                environment_class.__init__(self, **env_parameters)
                self.state_visitations = [[0 for _ in range(num_states)] for _ in range(num_skills)]
                self.regularisation_weight = regularisation_weight
                self.visitations_decay = visitations_decay

            def print_state_distribution(self):
                print(self.state_visitations)
                state_count = {k: 0 for k in range(num_states)}
                for skill in range(len(self.state_visitations)):
                    for state in range(len(self.state_visitations[0])):
                        state_count[state] += self.state_visitations[skill][state]
                probability_visitations = [[row[ix] / state_count[ix] for ix in range(len(row))] for row in
                                           self.state_visitations]
                print(" ")
                print(probability_visitations)
                print(" ")

            def reset(self):
                self.skill = random.randint(0, num_skills - 1)  # randomly choose among skills
                environment_class.reset(self)
                return np.array([self.state, self.skill])

            def step(self, action):
                next_state, reward, done, _ = environment_class.step(self, action)
                self.update_state_visitations(next_state)
                probability_correct_skill = self.calculate_probability_correct_skill(next_state)
                new_reward = reward  - self.regularisation_weight * (1.0 / np.log(probability_correct_skill))
                return np.array([next_state, self.skill]), new_reward, done, _

            def calculate_probability_correct_skill(self, next_state):
                """Calculates the probability that a certain"""
                visitations_correct_skill = self.state_visitations[self.skill][next_state]
                visitations_any_skill = np.sum([visit[next_state] for visit in self.state_visitations])
                probability = float(visitations_correct_skill) / float(visitations_any_skill)
                return probability

            def update_state_visitations(self, next_state):
                self.state_visitations = [[val * self.visitations_decay for val in sublist] for sublist in
                                          self.state_visitations]
                self.state_visitations[self.skill][next_state] += 1
        return skills_env()

    def run_n_episodes(self):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        start = time.time()
        self.skill_agent_config.environment = self.create_skill_learning_environment(self.num_skills, self.skill_agent_config.hyperparameters["regularisation_weight"],
                                               self.skill_agent_config.hyperparameters["visitations_decay"], self.skill_agent_config.env_parameters,
                                               self.environment.observation_space.n)
        self.skill_agent = DQN(self.skill_agent_config)
        print("Pretraining Starting")
        self.skill_agent.run_n_episodes()
        print("Pretraining Finished")
        print("Final state distribution:")
        self.skill_agent.environment.print_state_distribution()

        self.skill_agent.turn_off_all_exploration()
        self.manager_config.environment = self.create_manager_learning_environment(self.manager_config.env_parameters, self.skill_agent,
                                                                                   self.timesteps_before_changing_skill, self.num_skills)
        self.manager_agent = DQN(self.manager_config)
        self.manager_agent.run_n_episodes()
        time_taken = time.time() - start
        return self.manager_agent.game_full_episode_scores, self.manager_agent.rolling_results, time_taken

    def create_manager_learning_environment(self, env_parameters, skills_agent, timesteps_before_changing_skill, num_skills):
        """Creates the environment for the manager to learn in after skills network is frozen"""
        environment_class = self.environment.__class__

        class manager_env(environment_class):
            """Creates an environment from within which to train the manager"""

            def __init__(self):  # , meta_agent):
                environment_class.__init__(self, **env_parameters)
                self.action_space = namedtuple('action_space', 'n dtype')
                self.action_space.n = num_skills
                self.action_space.dtype = int

            def step(self, skill):
                """Moves a step in manager environment which involves committing to using a skill for a set number of timesteps"""
                next_state = self.state
                cumulative_reward = 0
                for _ in range(timesteps_before_changing_skill):
                    skill_action = skills_agent.pick_action(np.array([next_state, skill]))
                    next_state, reward, done, _ = environment_class.step(self, skill_action)
                    cumulative_reward += reward
                    if done: break
                return next_state, cumulative_reward, done, _

        return manager_env()

