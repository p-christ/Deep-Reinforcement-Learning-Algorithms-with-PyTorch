import copy
import random
import time
from operator import itemgetter

import numpy as np
import torch

from torch import optim
from collections import Counter
from Base_Agent import Base_Agent
from Replay_Buffer import Replay_Buffer
from Utility_Functions import flatten_action_id_to_actions
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from k_Sequitur import k_Sequitur

class Model_HRL(Base_Agent):
    agent_name = "Model_HRL"

    def __init__(self, config):
        super().__init__(config)
        self.min_episode_score_seen = float("inf")
        self.end_of_episode_symbol = "/"
        self.grammar_calculator = k_Sequitur(k=config.hyperparameters["sequitur_k"], end_of_episode_symbol=self.end_of_episode_symbol)
        self.action_length_reward_bonus = self.hyperparameters["action_length_reward_bonus"]
        self.rolling_score = self.lowest_possible_episode_score
        self.episode_actions_scores_and_exploration_status = None
        self.episodes_to_run_with_no_exploration = self.hyperparameters["episodes_to_run_with_no_exploration"]
        self.use_global_list_of_best_performing_actions = self.hyperparameters["use_global_list_of_best_performing_actions"]
        self.global_action_id_to_primitive_action = {k: tuple([k]) for k in range(self.action_size)}
        self.num_top_results_to_use = self.hyperparameters["num_top_results_to_use"]
        self.global_list_of_best_results = []
        self.new_actions_just_added = []
        self.action_frequency_required_in_top_results = self.hyperparameters["action_frequency_required_in_top_results"]
        self.reduce_macro_action_appearance_cutoff_throughout_training = self.hyperparameters["reduce_macro_action_appearance_cutoff_throughout_training"]
        self.episodes_per_round = self.hyperparameters["episodes_per_round"]
        self.action_id_to_stepping_stone_action_id = {}
        self.add_1_macro_action_at_a_time = self.hyperparameters["add_1_macro_action_at_a_time"]
        self.agent = DDQN_Wrapper(config, self.global_action_id_to_primitive_action, self.action_length_reward_bonus)

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        start = time.time()
        if num_episodes is None: num_episodes = self.config.num_episodes_to_run
        self.num_episodes = num_episodes
        self.episodes_conducted = 0
        self.grammar_induction_iteration = 1
        while self.episodes_conducted < self.num_episodes:
            self.play_new_episodes()
            self.generate_new_grammar()
            self.update_agent()
        final_actions_count = Counter(self.round_of_macro_actions)
        print("FINAL EPISODE SET ACTIONS COUNT ", final_actions_count)
        time_taken = time.time() - start
        return self.agent.game_full_episode_scores[:self.num_episodes], self.agent.rolling_results[:self.num_episodes], time_taken

    def play_new_episodes(self):
        """Plays a new set of episodes using the recently updated agent"""
        self.episode_actions_scores_and_exploration_status, self.round_of_macro_actions = \
            self.agent.run_n_episodes(num_episodes=self.episodes_per_round,
                                      episodes_to_run_with_no_exploration=self.episodes_to_run_with_no_exploration)
        self.episodes_conducted += len(self.episode_actions_scores_and_exploration_status)

    def generate_new_grammar(self):
        """Infers a new action grammar and updates the global action set"""
        actions_to_infer_grammar_from = self.pick_actions_to_infer_grammar_from(self.episode_actions_scores_and_exploration_status)
        num_actions_before = len(self.global_action_id_to_primitive_action)
        if self.infer_new_grammar: self.update_action_choices(actions_to_infer_grammar_from)
        else: print("NOT inferring new grammar because no better results found")
        self.new_actions_just_added = list(range(num_actions_before, num_actions_before + len(
            self.global_action_id_to_primitive_action) - num_actions_before))
        self.check_new_global_actions_valid()
        self.grammar_induction_iteration += 1

    def pick_actions_to_infer_grammar_from(self, episode_actions_scores_and_exploration_status):
        """Takes in data summarising the results of the latest games the agent played and then picks the actions from which
        we want to base the subsequent action grammar on"""
        episode_scores = [data[0] for data in episode_actions_scores_and_exploration_status]
        episode_actions = [data[1] for data in episode_actions_scores_and_exploration_status]
        reverse_ordering = np.argsort(episode_scores)
        top_result_indexes = list(reverse_ordering[-self.num_top_results_to_use:])
        best_episode_actions = list(itemgetter(*top_result_indexes)(episode_actions))
        best_episode_rewards = list(itemgetter(*top_result_indexes)(episode_scores))
        if self.use_global_list_of_best_performing_actions:
            best_result_this_round = max(best_episode_rewards)
            if len(self.global_list_of_best_results) == 0:
                worst_best_result_ever = float("-inf")
            else:
                worst_best_result_ever = min([data[0] for data in self.global_list_of_best_results])
            if best_result_this_round > worst_best_result_ever:
                self.keep_track_of_best_results_seen_so_far(best_episode_rewards, best_episode_actions)
                best_episode_actions = [data[1] for data in self.global_list_of_best_results]
                print("AFTER ", best_episode_actions)
                print("AFter best results ", [data[0] for data in self.global_list_of_best_results])
        best_episode_actions = [item for sublist in best_episode_actions for item in sublist]
        return best_episode_actions

    def keep_track_of_best_results_seen_so_far(self, top_results, best_episode_actions):
        """Keeps a track of the top episode results so far & the actions played in those episodes"""
        self.infer_new_grammar = True
        combined_result = [(result, actions) for result, actions in zip(top_results, best_episode_actions)]
        self.logger.info("New Candidate Best Results: {}".format(combined_result))
        self.global_list_of_best_results += combined_result
        self.global_list_of_best_results.sort(key=lambda x: x[0], reverse=True)
        self.global_list_of_best_results = self.global_list_of_best_results[:self.num_top_results_to_use]
        self.logger.info("After Best Results: {}".format(self.global_list_of_best_results))
        assert isinstance(self.global_list_of_best_results, list)
        assert isinstance(self.global_list_of_best_results[0], tuple)
        assert len(self.global_list_of_best_results[0]) == 2

    def update_action_choices(self, latest_macro_actions_seen):
        """Creates a grammar out of the latest list of macro actions conducted by the agent"""
        grammar_calculator = k_Sequitur(k=self.config.hyperparameters["sequitur_k"],
                                        end_of_episode_symbol=self.end_of_episode_symbol)
        print("latest_macro_actions_seen ", latest_macro_actions_seen)
        _, _, _, rules_episode_appearance_count = grammar_calculator.generate_action_grammar(latest_macro_actions_seen)
        print("NEW rules_episode_appearance_count ", rules_episode_appearance_count)
        new_actions = self.pick_new_macro_actions(rules_episode_appearance_count)
        self.update_global_action_id_to_primitive_action(new_actions)

    def pick_new_macro_actions(self, rules_episode_appearance_count):
        """Picks the new macro actions to be made available to the agent. Returns them in the form {action_id: (action_1, action_2, ...)}.
        NOTE there are many ways to do this... i should do experiments testing different ways and report the results
        """
        new_unflattened_actions = {}
        cutoff = self.num_top_results_to_use * self.action_frequency_required_in_top_results
        if self.reduce_macro_action_appearance_cutoff_throughout_training:
            cutoff = cutoff / (self.grammar_induction_iteration**0.5)
        print(" ")
        print("Cutoff ", cutoff)
        print(" ")
        action_id = len(self.global_action_id_to_primitive_action.keys())
        counts = {}
        for rule in rules_episode_appearance_count.keys():
            count = rules_episode_appearance_count[rule]
            # count = count * (len(rule))**0.25
            print("Rule {} -- Count {}".format(rule, count))
            if count >= cutoff:
                new_unflattened_actions[action_id] = rule
                counts[action_id] = count
                action_id += 1
        new_actions = flatten_action_id_to_actions(new_unflattened_actions, self.global_action_id_to_primitive_action,
                                                   self.action_size)
        if self.add_1_macro_action_at_a_time:
            max_count = 0
            best_rule = None
            for action_id, primitive_actions in new_actions.items():
                if primitive_actions not in self.global_action_id_to_primitive_action.values():
                    count = counts[action_id]
                    if count > max_count:
                        max_count = count
                        best_rule = primitive_actions
            if best_rule is None: new_actions = {}
            else:
                new_actions = {len(self.global_action_id_to_primitive_action.keys()): best_rule}
        return new_actions

    def update_global_action_id_to_primitive_action(self, new_actions):
        """Updates global_action_id_to_primitive_action by adding any new actions in that aren't already represented"""
        print("update_global_action_id_to_primitive_action ", new_actions)
        unique_new_actions = {k: v for k, v in new_actions.items() if v not in self.global_action_id_to_primitive_action.values()}
        next_action_name = max(self.global_action_id_to_primitive_action.keys()) + 1
        for _, value in unique_new_actions.items():
            self.global_action_id_to_primitive_action[next_action_name] = value
            # self.update_action_id_to_stepping_stone_action_id(next_action_name, value)
            next_action_name += 1

    def check_new_global_actions_valid(self):
        """Checks that global_action_id_to_primitive_action still only has valid entries"""
        assert len(set(self.global_action_id_to_primitive_action.values())) == len(
            self.global_action_id_to_primitive_action.values()), \
            "Not all actions are unique anymore: {}".format(self.global_action_id_to_primitive_action)
        for key, value in self.global_action_id_to_primitive_action.items():
            assert max(value) < self.action_size, "Actions should be in terms of primitive actions"

    def update_agent(self):
        """Updates the macro-actions that the agent can choose"""
        self.agent.global_action_id_to_primitive_actions = self.global_action_id_to_primitive_action
        print("New action set ", self.agent.global_action_id_to_primitive_actions)


