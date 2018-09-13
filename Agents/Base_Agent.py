import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from Memory_Data_Structures.Replay_Buffer import Replay_Buffer
from abc import ABC, abstractmethod

from Utilities import abstract

@abstract
class Base_Agent(object):    
    
    def __init__(self, environment, seed, hyperparameters, rolling_score_length, average_score_required,
                 agent_name):
        self.environment = environment        
        self.action_size = self.environment.get_action_size()
        self.state_size = self.environment.get_state_size()
        self.hyperparameters = hyperparameters

        self.rolling_score_length = rolling_score_length
        self.average_score_required = average_score_required
        self.total_episode_score_so_far = 0
        self.game_full_episode_scores = []
        self.rolling_results = []
        self.max_rolling_score_seen = float("-inf")
        self.agent_name = agent_name
        self.episode_number = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.environment.reset_environment()
        self.state = self.environment.get_state()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.step_number = 0

    def run_n_episodes(self, num_episodes_to_run=1, save_model=False):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        for episode in range(num_episodes_to_run):
            self.reset_game()
            self.episode_number += 1
            self.run_episode()
            self.save_and_print_result()          

            
            if self.max_rolling_score_seen > self.average_score_required: #stop once we achieve required score
                break
        
        self.summarise_results()
        if save_model:
            self.locally_save_policy()
        return self.game_full_episode_scores, self.rolling_results

    def run_episode(self):
        """Runs a full episode"""
        while not self.done:
                self.step()

    @abstractmethod
    def step(self):
        pass


    def conduct_action(self):
        self.environment.conduct_action(self.action)

        
    def update_next_state_reward_done_and_score(self): 
        self.next_state = self.environment.get_next_state()
        self.reward = self.environment.get_reward()
        self.done = self.environment.get_done()
        self.total_episode_score_so_far += self.environment.get_reward()


    @abstractmethod
    def time_to_learn(self):
        pass


    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def save_experience(self):
        pass

    def save_and_print_result(self):
        self.save_result()
        self.print_rolling_result()

    def save_result(self):
        self.game_full_episode_scores.append(self.total_episode_score_so_far)
        self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_length:]))
        self.save_max_result_seen()

    def save_max_result_seen(self):
        if self.rolling_results[-1] > self.max_rolling_score_seen:
            if len(self.rolling_results) > self.rolling_score_length:
                self.max_rolling_score_seen = self.rolling_results[-1]

    def print_rolling_result(self):
        sys.stdout.write(
            "\r Episode {0}, Rolling score: {1: .2f}, Max rolling score seen: {2: .2f}".format(len(self.game_full_episode_scores),
                                                                                               self.rolling_results[-1],
                                                                                               self.max_rolling_score_seen))
        sys.stdout.flush()

    def summarise_results(self):
        self.show_whether_achieved_goal()                                  
        self.visualise_results()

    def show_whether_achieved_goal(self):

        index_achieved_goal = self.achieved_required_score_at_index()
        print(" ")
        if index_achieved_goal == -1: #this means agent never achieved goal
            print("\033[91m" + "\033[1m" +
                  "{} did not achieve required score \n".format(self.agent_name) +
                  "\033[0m" + "\033[0m")
        else:
            print("\033[92m" + "\033[1m" +
                  "{} achieved required score at episode {} \n".format(self.agent_name, index_achieved_goal) +
                  "\033[0m" + "\033[0m")

    def achieved_required_score_at_index(self):
        """Returns the episode at which agent achieved goal or -1 if it never achieved it"""
        for ix, score in enumerate(self.rolling_results):
            if score > self.average_score_required:
                return ix
        return -1

    def visualise_results(self):
        plt.plot(self.rolling_results)
        plt.ylabel('Episode score')
        plt.xlabel('Episode number')
        plt.show()   

    @abstractmethod
    def locally_save_policy(self):
        pass


        

    



    
