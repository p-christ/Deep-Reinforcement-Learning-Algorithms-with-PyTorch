from operator import itemgetter
import numpy as np

from Utility_Functions import flatten_action_id_to_actions
from k_Sequitur import k_Sequitur


class Grammar_Generator(object):
    """Takes as input the actions in the best performing episodes and prdocues an updated action list"""

    def __init__(self, num_top_results_to_use, action_size, add_1_macro_action_at_a_time, use_relative_counts,
                 reduce_macro_action_appearance_cutoff_throughout_training, logger, sequitur_k,  action_frequency_required_in_top_results,
                 end_of_episode_symbol = "/"):
        self.logger = logger
        self.global_list_of_best_results = []
        self.global_list_of_bad_results = []
        self.num_top_results_to_use = num_top_results_to_use
        self.action_size = action_size
        self.add_1_macro_action_at_a_time = add_1_macro_action_at_a_time
        self.reduce_macro_action_appearance_cutoff_throughout_training = reduce_macro_action_appearance_cutoff_throughout_training
        self.use_relative_counts = use_relative_counts
        self.end_of_episode_symbol = end_of_episode_symbol
        self.sequitur_k = sequitur_k
        self.action_frequency_required_in_top_results = action_frequency_required_in_top_results

    def generate_new_grammar(self, episode_actions_scores_and_exploration_status, global_action_id_to_primitive_action):
        """Infers a new action grammar and updates the global action set"""
        self.action_id_to_action = global_action_id_to_primitive_action
        good_actions, bad_actions = self.pick_actions_to_infer_grammar_from(episode_actions_scores_and_exploration_status)
        num_actions_before = len(global_action_id_to_primitive_action)

        if len(bad_actions) > 0:
            self.update_action_choices(good_actions, bad_actions)
            self.check_new_global_actions_valid()
        new_actions_just_added = list(range(num_actions_before, num_actions_before + len(
             global_action_id_to_primitive_action) - num_actions_before))
        return self.action_id_to_action, new_actions_just_added

    def pick_actions_to_infer_grammar_from(self, episode_actions_scores_and_exploration_status):
        """Takes in data summarising the results of the latest games the agent played and then picks the actions from which
        we want to base the subsequent action grammar on"""
        episode_scores = [data[0] for data in episode_actions_scores_and_exploration_status]
        episode_actions = [data[1] for data in episode_actions_scores_and_exploration_status]
        reverse_ordering = np.argsort(episode_scores)
        top_result_indexes = list(reverse_ordering[-self.num_top_results_to_use:])

        best_episode_actions = list(itemgetter(*top_result_indexes)(episode_actions))
        best_episode_rewards = list(itemgetter(*top_result_indexes)(episode_scores))

        best_result_this_round = max(best_episode_rewards)
        if len(self.global_list_of_best_results) == 0:
            worst_best_result_ever = float("-inf")
        else:
            worst_best_result_ever = min([data[0] for data in self.global_list_of_best_results])
        if best_result_this_round > worst_best_result_ever:
            combined_best_results = [(result, actions) for result, actions in zip( best_episode_rewards, best_episode_actions)]
            self.global_list_of_best_results, new_old_best_results = self.keep_track_of_best_results_seen_so_far(
                self.global_list_of_best_results, combined_best_results)
            self.global_list_of_bad_results, _ = self.keep_track_of_best_results_seen_so_far(
                self.global_list_of_bad_results, new_old_best_results)

        best_episode_actions = [data[1] for data in self.global_list_of_best_results]
        bad_episode_actions = [data[1] for data in self.global_list_of_bad_results]
        print("AFTER ", best_episode_actions)
        print("AFter best results ", [data[0] for data in self.global_list_of_best_results])
        print("AFter bad results ", [data[0] for data in self.global_list_of_bad_results])

        best_episode_actions = [item for sublist in best_episode_actions for item in sublist]
        bad_episode_actions = [item for sublist in bad_episode_actions for item in sublist]
        return best_episode_actions, bad_episode_actions

    def keep_track_of_best_results_seen_so_far(self, global_results, combined_result):
        """Keeps a track of the top episode results so far & the actions played in those episodes"""
        self.logger.info("New Candidate Best Results: {}".format(combined_result))
        print(combined_result)
        global_results += combined_result
        global_results.sort(key=lambda x: x[0], reverse=True)
        global_results, old_best_results = global_results[:self.num_top_results_to_use], global_results[self.num_top_results_to_use:]
        assert isinstance(global_results, list)
        if len(global_results) > 0:
            assert isinstance(global_results[0], tuple)
            assert len(global_results[0]) == 2
        return global_results, old_best_results



    def check_new_global_actions_valid(self):
        """Checks that global_action_id_to_primitive_action still only has valid entries"""
        assert len(set(self.action_id_to_action.values())) == len(
            self.action_id_to_action.values()), \
            "Not all actions are unique anymore: {}".format(self.action_id_to_action)
        for key, value in self.action_id_to_action.items():
            assert max(value) < self.action_size, "Actions should be in terms of primitive actions"


    def update_action_choices(self, good_actions, bad_actions):
        """Creates a grammar out of the latest list of macro actions conducted by the agent"""
        good_episode_rule_appearance = self.get_rule_appearance_count(good_actions)
        bad_episode_rule_appearance = self.get_rule_appearance_count(bad_actions)
        if not self.use_relative_counts:
            new_actions = self.pick_new_macro_actions(good_episode_rule_appearance)
        else:
            new_actions = self.pick_new_macro_actions_using_relative_data(good_episode_rule_appearance, bad_episode_rule_appearance)
        self.update_global_action_id_to_primitive_action(new_actions)

    def get_rule_appearance_count(self, actions):
        """Takes as input a list of actions and infers rules using Sequitur and then returns a dictionary indicating how
        many times each rule was used"""
        grammar_calculator = k_Sequitur(k=self.sequitur_k,
                                        end_of_episode_symbol=self.end_of_episode_symbol)
        print("latest_macro_actions_seen ", actions)
        _, _, _, rules_episode_appearance_count = grammar_calculator.generate_action_grammar(actions)
        print("NEW rules_episode_appearance_count ", rules_episode_appearance_count)
        return rules_episode_appearance_count

    def update_global_action_id_to_primitive_action(self, new_actions):
        """Updates global_action_id_to_primitive_action by adding any new actions in that aren't already represented"""
        print("update_global_action_id_to_primitive_action ", new_actions)
        unique_new_actions = {k: v for k, v in new_actions.items() if v not in self.action_id_to_action.values()}
        next_action_name = max(self.action_id_to_action.keys()) + 1
        for _, value in unique_new_actions.items():
            self.action_id_to_action[next_action_name] = value
            next_action_name += 1

    def pick_new_macro_actions_using_relative_data(self, good_episode_rule_appearance, bad_episode_rule_appearance):
        """Picks new macro actions according to the macro actions common in current best episodes but not in previously best episodes"""
        new_unflattened_actions = {}
        cutoff = self.calculate_cutoff_for_macro_actions()
        counts = {}
        action_id = len(self.action_id_to_action.keys())

        for rule in good_episode_rule_appearance.keys():
            good_count = good_episode_rule_appearance[rule]
            if rule in bad_episode_rule_appearance.keys():
                bad_count = bad_episode_rule_appearance[rule]
            else:
                bad_count = 0
            count_difference = good_count - bad_count
            print("Rule {} -- Count Difference {}".format(rule, count_difference))
            if count_difference >= cutoff:
                new_unflattened_actions[action_id] = rule
                counts[action_id] = count_difference
                action_id += 1
        new_actions = flatten_action_id_to_actions(new_unflattened_actions, self.action_id_to_action,
                                                   self.action_size)
        if self.add_1_macro_action_at_a_time:
            new_actions = self.find_highest_count_macro_action(new_actions, counts)
        return new_actions

    def calculate_cutoff_for_macro_actions(self):
        """Calculates how many times a macro action needs to appear before we include it in the action set"""
        cutoff = self.num_top_results_to_use * self.action_frequency_required_in_top_results
        if self.reduce_macro_action_appearance_cutoff_throughout_training:
            cutoff = cutoff / (self.grammar_induction_iteration ** 0.5)
        print(" ")
        print("Cutoff ", cutoff)
        print(" ")
        return cutoff

    def pick_new_macro_actions(self, rules_episode_appearance_count):
        """Picks the new macro actions to be made available to the agent. Returns them in the form {action_id: (action_1, action_2, ...)}.
        NOTE there are many ways to do this... i should do experiments testing different ways and report the results
        """
        new_unflattened_actions = {}
        cutoff = self.calculate_cutoff_for_macro_actions()

        action_id = len(self.action_id_to_action.keys())
        counts = {}
        for rule in rules_episode_appearance_count.keys():
            count = rules_episode_appearance_count[rule]
            print("Rule {} -- Count {}".format(rule, count))
            if count >= cutoff:
                new_unflattened_actions[action_id] = rule
                counts[action_id] = count
                action_id += 1
        new_actions = flatten_action_id_to_actions(new_unflattened_actions, self.action_id_to_action,
                                                   self.action_size)
        if self.add_1_macro_action_at_a_time:
            new_actions = self.find_highest_count_macro_action(new_actions, counts)
        return new_actions

    def find_highest_count_macro_action(self, new_actions, counts):
        """Finds macro count with highest count"""
        max_count = 0
        best_rule = None
        for action_id, primitive_actions in new_actions.items():
            if primitive_actions not in self.action_id_to_action.values():
                count = counts[action_id]
                if count > max_count:
                    max_count = count
                    best_rule = primitive_actions
        if best_rule is None:
            new_actions = {}
        else:
            new_actions = {len(self.action_id_to_action.keys()): best_rule}
        return new_actions
