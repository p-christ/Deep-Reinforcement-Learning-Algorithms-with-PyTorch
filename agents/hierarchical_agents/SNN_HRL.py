import copy
import random
import time
import numpy as np
import torch
from gym import Wrapper, spaces
from agents.Base_Agent import Base_Agent
from agents.policy_gradient_agents.PPO import PPO
from agents.DQN_agents.DDQN import DDQN


class SNN_HRL(Base_Agent):
    """Implements the hierarchical RL agent that uses stochastic neural networks (SNN) from the paper Florensa et al. 2017
    https://arxiv.org/pdf/1704.03012.pdf
    Works by:
    1) Creating a pre-training environment within which the skill_agent can learn for some period of time
    2) Then skill_agent is frozen
    3) Then we train a manager agent that chooses which of the pre-trained skills to let act for it for some period of time
    Note that it only works with discrete states at the moment.

    Note that this agent will not work well in environments where it is beneficial to end the game as quickly as possible
    because then there isn't enough incentive for the skills to learn to explore different parts of the state space
    """
    agent_name = "SNN-HRL"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert isinstance(self.environment.reset(), int) or isinstance(self.environment.reset(), np.int64) or  self.environment.reset().dtype == np.int64, "only works for discrete states currently"
        self.num_skills = self.hyperparameters["SKILL_AGENT"]["num_skills"]
        self.episodes_for_pretraining =  self.hyperparameters["SKILL_AGENT"]["episodes_for_pretraining"]
        self.timesteps_before_changing_skill = self.hyperparameters["MANAGER"]["timesteps_before_changing_skill"]

        self.skill_agent_config = copy.deepcopy(config)
        self.skill_agent_config.hyperparameters = self.skill_agent_config.hyperparameters["SKILL_AGENT"]
        self.skill_agent_config.num_episodes_to_run = self.episodes_for_pretraining

        self.manager_config = copy.deepcopy(config)
        self.manager_config.hyperparameters = self.manager_config.hyperparameters["MANAGER"]
        self.manager_config.num_episodes_to_run = self.config.num_episodes_to_run - self.skill_agent_config.num_episodes_to_run

    def run_n_episodes(self):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        start = time.time()

        skill_agent = self.create_skill_training_agent()
        skill_agent.run_n_episodes()
        self.skill_agent_config.environment.print_state_distribution()
        skill_agent.turn_off_any_epsilon_greedy_exploration()

        manager_agent = self.create_manager_agent(skill_agent)
        manager_agent.run_n_episodes()

        time_taken = time.time() - start
        pretraining_results = [np.min(manager_agent.game_full_episode_scores)]*self.episodes_for_pretraining
        return pretraining_results + manager_agent.game_full_episode_scores, pretraining_results + manager_agent.rolling_results, time_taken

    def create_skill_training_agent(self):
        """Creates and instantiates a pre-training environment for the agent to learn skills in and then instantiates
        and agent to learn in this environment"""
        self.skill_agent_config.environment = Skill_Wrapper(copy.deepcopy(self.environment), self.environment.observation_space.n,
                                                            self.num_skills,
                                                            self.skill_agent_config.hyperparameters[
                                                                "regularisation_weight"], self.skill_agent_config.hyperparameters["visitations_decay"])
        return DDQN(self.skill_agent_config)

    def create_manager_agent(self, skill_agent):
        """Instantiates a manager agent"""
        self.manager_config.environment = Manager_Frozen_Worker_Wrapper(copy.deepcopy(self.environment), self.num_skills,
                                                                             self.timesteps_before_changing_skill, skill_agent)
        return DDQN(self.manager_config)


class Skill_Wrapper(Wrapper):
    """Open AI gym wrapper to help create a pretraining environment in which to train skills"""
    def __init__(self, env, num_states, num_skills, regularisation_weight, visitations_decay):
        Wrapper.__init__(self, env)
        self.num_skills = num_skills
        self.num_states = num_states
        self.state_visitations = [[0 for _ in range(num_states)] for _ in range(num_skills)]
        self.regularisation_weight = regularisation_weight
        self.visitations_decay = visitations_decay

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.skill = random.randint(0, self.num_skills - 1)
        return self.observation(observation)

    def observation(self, observation):
        return np.concatenate((np.array(observation), np.array([self.skill])))

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        new_reward = self.calculate_new_reward(reward, next_state)
        return self.observation(next_state), new_reward, done, _

    def calculate_new_reward(self, reward, next_state):
        self.update_state_visitations(next_state)
        probability_correct_skill = self.calculate_probability_correct_skill(next_state)
        new_reward = reward + self.regularisation_weight * np.log(probability_correct_skill)
        return new_reward

    def update_state_visitations(self, next_state):
        """Updates table keeping track of number of times each state visited under each skill"""
        self.state_visitations = [[val * self.visitations_decay for val in sublist] for sublist in
                                  self.state_visitations]
        self.state_visitations[self.skill][next_state[0]] += 1

    def calculate_probability_correct_skill(self, next_state):
        """Calculates the probability that being in a state implies a certain skill"""
        visitations_correct_skill = self.state_visitations[self.skill][next_state[0]]
        visitations_any_skill = np.sum([visit[next_state[0]] for visit in self.state_visitations])
        probability = float(visitations_correct_skill) / float(visitations_any_skill)
        return probability

    def print_state_distribution(self):
        """Prints the observed probability of skills depending on the state we are in"""
        print(self.state_visitations)
        state_count = {k: 0 for k in range(self.num_states)}
        for skill in range(len(self.state_visitations)):
            for state in range(len(self.state_visitations[0])):
                state_count[state] += self.state_visitations[skill][state]
        probability_visitations = [[row[ix] / max(1.0, state_count[ix]) for ix in range(len(row))] for row in
                                   self.state_visitations]
        print(" ")
        print(probability_visitations)
        print(" ")

class Manager_Frozen_Worker_Wrapper(Wrapper):
    """Open AI gym wrapper to help create an environment where manager learns to act by instructing a frozen worker"""
    def __init__(self, env, num_skills, timesteps_before_changing_skill, skills_agent):
        Wrapper.__init__(self, env)
        self.action_space = spaces.Discrete(num_skills)
        self.timesteps_before_changing_skill = timesteps_before_changing_skill
        self.skills_agent = skills_agent

    def step(self, action):
        """Moves a step in manager environment which involves committing to using a skill for a set number of timesteps"""
        next_state = self.env.unwrapped.s
        cumulative_reward = 0
        for _ in range(self.timesteps_before_changing_skill):
            with torch.no_grad():
                skill_action = self.skills_agent.pick_action(np.array([next_state[0], action]))
            next_state, reward, done, _ = self.env.step(skill_action)
            cumulative_reward += reward
            if done: break
        return next_state, cumulative_reward, done, _