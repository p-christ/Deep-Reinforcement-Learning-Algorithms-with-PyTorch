import random
from collections import Counter

from gym import Wrapper, spaces
from torch import nn, optim
from Base_Agent import Base_Agent
import copy
import time
import numpy as np
from DDQN import DDQN
from Memory_Shaper import Memory_Shaper
from Utility_Functions import flatten_action_id_to_actions
from k_Sequitur import k_Sequitur
import numpy as np
from operator import itemgetter

# Presentation ideas:
# 1) Show that agent at end is using macro actions and not ignoring them
# 2) Show an example final episode using a macro action
# 3) Use 10 random seeds to compare algos
# 4) Compare DDQN with this algo
# 5) Point is this game too simple to benefit from macro actions but it is using them
# 6) Use name Hindsight Macro-Action Experience Replay


# TODO train model to predict next state too so that we can use this to figure out when to abandon macro actions
# TODO do grammar updates less often as we go...  increase % improvement required as we go?
# TODO bias grammar inference to be bias towards longer actions
# TODO don't base grammar updates on progress
# TODO do pre-training until loss stops going down quickly?
# TODO Use SAC-Discrete instead of DQN?
# TODO learn grammar of a mixture of best performing episodes when had exploration on vs. not
# TODO higher learning rate when just training final layer?
# TODO write a check that the final layers of network learn properly after we change them
# TODO add mechanism to go backwards if we picked an earlier macro action that then becomes irrelevant?
# TODO change so its the action rules used in most of best performing episodes that get used rather than
#      those that occur the most overall. because a better episode ends faster and so less occurances of actions!!
#      maybe even pick out the fixed X many actions from the top performing episodes no matter how many episodes
# TODO have higher minimum bound for number episodes to retrain on
# TODO try starting with all the 2 step moves as macro actions...  or even starting with random macro actions?
# TODO try having the threshold for how often need see actions come down throughout training?
# TODO fix the tests for memory shaper... very important this works properly
# TODO try just learning off the top 2 episodes? instead of top 10?
# TODO pick actions that are most common in best episodes but also not common in the less well preforming episodes
# TODO could refresh the macro actions we pick later in game
# TODO add option for full random play for some episodes
# TODO try a constant exploration rate
# TODO have DQN agent using an action balanced replay buffer from START not just from 1st grammar iteration?
# TODO try where number of pre-training iterations is fixed and doesnt depend on number of new actions
# TODO try adding 1 action at a time?
# TODO make multi action q-values more self consistent by forcing them to be the first actions q-values + some value
# TODO use a fixed leraning rate when training new actions rather than a learning rate that is lower because we near required score
# TODO add option for using no_grad when combining actions values from different places


class HRL(Base_Agent):
    agent_name = "HRL"

    def __init__(self, config):
        super().__init__(config)
        self.min_episode_score_seen = float("inf")
        self.end_of_episode_symbol = "/"
        self.grammar_calculator = k_Sequitur(k=config.hyperparameters["sequitur_k"], end_of_episode_symbol=self.end_of_episode_symbol)
        self.memory_shaper = Memory_Shaper(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"], config.seed,
                                           self.update_reward_to_encourage_longer_macro_actions)
        self.action_length_reward_bonus = self.hyperparameters["action_length_reward_bonus"]

        self.rolling_score = self.lowest_possible_episode_score
        self.episode_actions_scores_and_exploration_status = None
        self.episodes_to_run_with_no_exploration = self.hyperparameters["episodes_to_run_with_no_exploration"]
        self.pre_training_learning_iterations_multiplier = self.hyperparameters["pre_training_learning_iterations_multiplier"]
        self.copy_over_hidden_layers = self.hyperparameters["copy_over_hidden_layers"]
        self.use_global_list_of_best_performing_actions = self.hyperparameters["use_global_list_of_best_performing_actions"]

        self.global_action_id_to_primitive_action = {k: tuple([k]) for k in range(self.action_size)}
        self.num_top_results_to_use = self.hyperparameters["num_top_results_to_use"]

        self.agent = DDQN_Wrapper(config, self.global_action_id_to_primitive_action,
                             self.update_reward_to_encourage_longer_macro_actions, self.memory_shaper)

        self.global_list_of_best_results = []
        self.new_actions_just_added = []

        self.action_frequency_required_in_top_results = self.hyperparameters["action_frequency_required_in_top_results"]

        self.only_train_new_actions = self.hyperparameters["only_train_new_actions"]
        self.only_train_final_layer = self.hyperparameters["only_train_final_layer"]
        self.reduce_macro_action_appearance_cutoff_throughout_training = self.hyperparameters["reduce_macro_action_appearance_cutoff_throughout_training"]
        self.add_1_macro_action_at_a_time = self.hyperparameters["add_1_macro_action_at_a_time"]
        self.calculate_q_values_as_increments = self.hyperparameters["calculate_q_values_as_increments"]

        self.min_num_episodes_to_play = self.hyperparameters["min_num_episodes_to_play"]

        self.action_id_to_stepping_stone_action_id = {}

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):

        start = time.time()

        if num_episodes is None: num_episodes = self.config.num_episodes_to_run
        self.num_episodes = num_episodes
        self.episodes_conducted = 0
        self.grammar_induction_iteration = 1

        while self.episodes_conducted < self.num_episodes:

            self.episode_actions_scores_and_exploration_status, round_of_macro_actions = \
                self.agent.run_n_episodes(num_episodes=self.calculate_how_many_episodes_to_play(),
                                          episodes_to_run_with_no_exploration=self.episodes_to_run_with_no_exploration)


            self.episodes_conducted += len(self.episode_actions_scores_and_exploration_status)
            actions_to_infer_grammar_from = self.pick_actions_to_infer_grammar_from(
                self.episode_actions_scores_and_exploration_status)

            num_actions_before = len(self.global_action_id_to_primitive_action)

            if self.infer_new_grammar:
                self.update_action_choices(actions_to_infer_grammar_from)

            else:
                print("NOT inferring new grammar because no better results found")

            print("New actions ", self.global_action_id_to_primitive_action)

            self.new_actions_just_added = list(range(num_actions_before, num_actions_before + len(self.global_action_id_to_primitive_action) - num_actions_before))
            print("Actions just added ", self.new_actions_just_added)

            assert len(set(self.global_action_id_to_primitive_action.values())) == len(
                self.global_action_id_to_primitive_action.values()), \
                "Not all actions are unique anymore: {}".format(self.global_action_id_to_primitive_action)

            for key, value in self.global_action_id_to_primitive_action.items():
                assert max(value) < self.action_size, "Actions should be in terms of primitive actions"

            self.grammar_induction_iteration += 1

            current_num_actions = len(self.global_action_id_to_primitive_action.keys())

            if self.only_train_new_actions:
                PRE_TRAINING_ITERATIONS = int(self.pre_training_learning_iterations_multiplier * (len(self.new_actions_just_added) ** 1.25))
            else:
                PRE_TRAINING_ITERATIONS = int(self.pre_training_learning_iterations_multiplier * (current_num_actions ** 1.25))

            print(" ")

            print("PRE TRAINING ITERATIONS ", PRE_TRAINING_ITERATIONS)

            print(" ")

            self.agent.update_agent_for_new_actions(self.global_action_id_to_primitive_action,
                                                    copy_over_hidden_layers=self.copy_over_hidden_layers,
                                                    change_or_append_final_layer="APPEND")

            if num_actions_before != len(self.global_action_id_to_primitive_action):
                replay_buffer = self.memory_shaper.put_adapted_experiences_in_a_replay_buffer(
                    self.global_action_id_to_primitive_action)

                print(" ------ ")
                print("Length of buffer {} -- Actions {} -- Pre training iterations {}".format(len(replay_buffer),
                                                                                               current_num_actions,
                                                                                               PRE_TRAINING_ITERATIONS))
                print(" ------ ")
                self.overwrite_replay_buffer_and_pre_train_agent(replay_buffer, PRE_TRAINING_ITERATIONS,
                                                                 only_train_final_layer=self.only_train_final_layer, only_train_new_actions=self.only_train_new_actions)
            print("Now there are {} actions: {}".format(current_num_actions, self.global_action_id_to_primitive_action))


        episode_actions = [data[1] for data in self.episode_actions_scores_and_exploration_status]
        flat_episode_actions = [ep_action for ep in episode_actions for ep_action in ep]

        final_actions_count = Counter(round_of_macro_actions)
        print("FINAL EPISODE SET ACTIONS COUNT ", final_actions_count)


        time_taken = time.time() - start

        return self.agent.game_full_episode_scores[:self.num_episodes], self.agent.rolling_results[:self.num_episodes], time_taken

    def calculate_how_many_episodes_to_play(self):
        """Calculates how many episodes the agent should play until we re-infer the grammar"""
        episodes_to_play = self.hyperparameters["epsilon_decay_rate_denominator"] / self.grammar_induction_iteration
        episodes_to_play = max(self.min_num_episodes_to_play, int(max(self.episodes_to_run_with_no_exploration * 2, episodes_to_play)))
        print("Grammar iteration {} -- Episodes to play {}".format(self.grammar_induction_iteration, episodes_to_play))
        return episodes_to_play

    def keep_track_of_best_results_seen_so_far(self, top_results, best_episode_actions):
        """Keeps a track of the top episode results so far & the actions played in those episodes"""
        self.infer_new_grammar = True

        # old_result = copy.deepcopy(self.global_list_of_best_results)

        combined_result = [(result, actions) for result, actions in zip(top_results, best_episode_actions)]
        self.logger.info("New Candidate Best Results: {}".format(combined_result))

        self.global_list_of_best_results += combined_result
        self.global_list_of_best_results.sort(key=lambda x: x[0], reverse=True)
        self.global_list_of_best_results = self.global_list_of_best_results[:self.num_top_results_to_use]

        self.logger.info("After Best Results: {}".format(self.global_list_of_best_results))

        assert isinstance(self.global_list_of_best_results, list)
        assert isinstance(self.global_list_of_best_results[0], tuple)
        assert len(self.global_list_of_best_results[0]) == 2

        # Removing this because we want it to recaclculate grammar anyways even if top scores didn't change because our criteria
        # for grammar change now gets looser over time...
        # if old_result == self.global_list_of_best_results:
        #     self.infer_new_grammar = False

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
            self.keep_track_of_best_results_seen_so_far(best_episode_rewards, best_episode_actions)
            best_episode_actions = [data[1] for data in self.global_list_of_best_results]
            print("AFTER ", best_episode_actions)
            print("AFter best results ", [data[0] for data in self.global_list_of_best_results])

        best_episode_actions = [item for sublist in best_episode_actions for item in sublist]
        return best_episode_actions

    def update_action_choices(self, latest_macro_actions_seen):
        """Creates a grammar out of the latest list of macro actions conducted by the agent"""
        grammar_calculator = k_Sequitur(k=self.config.hyperparameters["sequitur_k"],
                                        end_of_episode_symbol=self.end_of_episode_symbol)
        print("latest_macro_actions_seen ", latest_macro_actions_seen)
        _, _, _, rules_episode_appearance_count = grammar_calculator.generate_action_grammar(latest_macro_actions_seen)
        print("NEW rules_episode_appearance_count ", rules_episode_appearance_count)
        new_actions = self.pick_new_macro_actions(rules_episode_appearance_count)
        self.update_global_action_id_to_primitive_action(new_actions)


    def update_global_action_id_to_primitive_action(self, new_actions):
        """Updates global_action_id_to_primitive_action by adding any new actions in that aren't already represented"""
        print("update_global_action_id_to_primitive_action ", new_actions)
        unique_new_actions = {k: v for k, v in new_actions.items() if v not in self.global_action_id_to_primitive_action.values()}
        next_action_name = max(self.global_action_id_to_primitive_action.keys()) + 1
        for _, value in unique_new_actions.items():
            self.global_action_id_to_primitive_action[next_action_name] = value
            # self.update_action_id_to_stepping_stone_action_id(next_action_name, value)
            next_action_name += 1


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

    def overwrite_replay_buffer_and_pre_train_agent(self, replay_buffer, training_iterations, only_train_final_layer,
                                                    only_train_new_actions):
        """Overwrites the replay buffer of the agent and sets it to the provided replay_buffer. Then trains the agent
        for training_iterations number of iterations using data from the replay buffer"""
        assert replay_buffer is not None
        self.agent.memory = replay_buffer
        if only_train_final_layer:
            print("Only training the final layer")
            self.freeze_all_but_output_layers(self.agent.q_network_local)
        for _ in range(training_iterations):
            if only_train_new_actions: new_actions = self.new_actions_just_added
            else: new_actions = []
            output = self.agent.learn(print_loss=False, only_these_actions=new_actions)
            if output == "BREAK": break
        if only_train_final_layer: self.unfreeze_all_layers(self.agent.q_network_local)

    def update_reward_to_encourage_longer_macro_actions(self, cumulative_reward, length_of_macro_action):
        """Update reward to encourage usage of longer macro actions. The size of the improvement depends positively
        on the length of the macro action"""
        if cumulative_reward == 0.0: increment = 0.1
        else: increment = abs(cumulative_reward)

        total_change = increment * ((length_of_macro_action - 1)** 0.5) * self.action_length_reward_bonus

        cumulative_reward += total_change
        return cumulative_reward

class DDQN_Wrapper(DDQN):

    def __init__(self, config, action_id_to_primitive_actions, bonus_reward_function,
                 memory_shaper, end_of_episode_symbol="/"):
        super().__init__(config)
        self.min_episode_score_seen = float("inf")
        self.end_of_episode_symbol = end_of_episode_symbol
        self.action_id_to_primitive_actions = action_id_to_primitive_actions
        self.bonus_reward_function = bonus_reward_function
        self.memory_shaper = memory_shaper
        self.action_id_to_stepping_stone_action_id = {}

    def update_agent_for_new_actions(self, action_id_to_primitive_actions, copy_over_hidden_layers, change_or_append_final_layer):
        assert change_or_append_final_layer in ["CHANGE", "APPEND"]
        num_actions_before = self.action_size
        self.action_id_to_primitive_actions = action_id_to_primitive_actions
        self.action_size = len(action_id_to_primitive_actions)
        num_new_actions = self.action_size - num_actions_before
        if num_new_actions > 0:
            for new_action_id in range(num_actions_before, num_actions_before + num_new_actions):
                self.update_action_id_to_stepping_stone_action_id(new_action_id)
            if change_or_append_final_layer == "CHANGE": self.change_final_layer_q_network(copy_over_hidden_layers)
            else: self.append_to_final_layers(num_new_actions)

    def update_action_id_to_stepping_stone_action_id(self, new_action_id):
        """Update action_id_to_stepping_stone_action_id with the new actions created"""
        new_action = self.action_id_to_primitive_actions[new_action_id]
        length_macro_action = len(new_action)
        print(" update_action_id_to_stepping_stone_action_id ")
        for sub_action_length in reversed(range(1, length_macro_action)):
            sub_action = new_action[:sub_action_length]
            if sub_action in self.action_id_to_primitive_actions.values():
                sub_action_id = list(self.action_id_to_primitive_actions.keys())[
                    list(self.action_id_to_primitive_actions.values()).index(sub_action)]

                self.action_id_to_stepping_stone_action_id[new_action_id] = sub_action_id
                print("Action {} has largest sub action {}".format(new_action_id, sub_action_id))
                break

    def append_to_final_layers(self, num_new_actions):
        """Appends to the end of a network to allow it to choose from the new actions. It does not change the weights
        for the other actions"""
        print("Appending options to final layer")
        assert num_new_actions > 0
        self.q_network_local.output_layers.append(nn.Linear(in_features=self.q_network_local.output_layers[0].in_features,
                                                            out_features=num_new_actions))
        self.q_network_target.output_layers.append(nn.Linear(in_features=self.q_network_local.output_layers[0].in_features,
                                                            out_features=num_new_actions))
        Base_Agent.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyperparameters["learning_rate"])

    def change_final_layer_q_network(self, copy_over_hidden_layers):
        """Completely changes the final layer of the q network to accomodate the new action space"""
        print("Completely changing final layer")
        assert len(self.q_network_local.output_layers) == 1
        if copy_over_hidden_layers:
            self.q_network_local.output_layers[0] = nn.Linear(in_features=self.q_network_local.output_layers[0].in_features,
                                                              out_features=self.action_size)
            self.q_network_target.output_layers[0] = nn.Linear(in_features=self.q_network_target.output_layers[0].in_features,
                                                              out_features=self.action_size)
        else:
            self.q_network_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
            self.q_network_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        Base_Agent.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyperparameters["learning_rate"])

    def run_n_episodes(self, num_episodes, episodes_to_run_with_no_exploration):
        self.turn_on_any_epsilon_greedy_exploration()

        self.round_of_macro_actions = []

        self.episode_actions_scores_and_exploration_status = []
        num_episodes_to_get_to = self.episode_number + num_episodes
        while self.episode_number < num_episodes_to_get_to:
            self.reset_game()
            self.step()
            self.save_and_print_result()
            if num_episodes_to_get_to - self.episode_number == episodes_to_run_with_no_exploration:
                self.turn_off_any_epsilon_greedy_exploration()

        assert len(self.episode_actions_scores_and_exploration_status) == num_episodes, "{} vs. {}".format(len(self.episode_actions_scores_and_exploration_status),
                                                                                                           num_episodes)
        assert len(self.episode_actions_scores_and_exploration_status[0]) == 3
        assert self.episode_actions_scores_and_exploration_status[0][2] in [True, False]
        assert isinstance(self.episode_actions_scores_and_exploration_status[0][1], list)
        assert isinstance(self.episode_actions_scores_and_exploration_status[0][1][0], int)
        assert isinstance(self.episode_actions_scores_and_exploration_status[0][0], int) or isinstance(self.episode_actions_scores_and_exploration_status[0][0], float)

        return self.episode_actions_scores_and_exploration_status, self.round_of_macro_actions


    def learn(self, experiences=None, print_loss=False, only_these_actions=[]):
        """Runs a learning iteration for the Q network"""
        if len(only_these_actions) == 0: super().learn()
        else:
            experiences = self.memory.sample_experiences_with_certain_actions(only_these_actions, self.action_size,
                                                                              self.hyperparameters["batch_size"])
            super().learn(experiences=experiences)

    def step(self):
        """Runs a step within a game including a learning step if required"""
        self.total_episode_score_so_far = 0
        macro_state = self.state
        state = self.state
        done = self.done

        episode_macro_actions = []

        while not done:
            macro_action = self.pick_action(state=macro_state)
            primitive_actions = self.action_id_to_primitive_actions[macro_action]
            macro_reward = 0
            primitive_actions_conducted = 0
            for action in primitive_actions:
                next_state, reward, done, _ = self.environment.step(action)
                macro_reward += reward
                self.total_episode_score_so_far += reward
                primitive_actions_conducted += 1
                self.track_episodes_data(state, action, reward, next_state, done)

                self.save_experience(experience=(state, action, reward, next_state, done))


                state = next_state
                if self.time_for_q_network_to_learn():
                    self.learn()
                if done or self.abandon_macro_action(): break

            macro_reward = self.bonus_reward_function(macro_reward, primitive_actions_conducted)
            macro_next_state = next_state
            macro_done = done
            if macro_action != action:
                self.save_experience(experience=(macro_state, macro_action, macro_reward, macro_next_state, macro_done))
            macro_state = macro_next_state

            episode_macro_actions.append(macro_action)
            self.round_of_macro_actions.append(macro_action)
        if random.random() < 0.1: print(Counter(episode_macro_actions))

        self.store_episode_in_memory_shaper()
        self.save_episode_actions_with_score()
        self.episode_number += 1

    def track_episodes_data(self, state, action, reward, next_state, done):
        self.episode_states.append(state)
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
        self.episode_next_states.append(next_state)
        self.episode_dones.append(done)

    def save_episode_actions_with_score(self):

        self.episode_actions_scores_and_exploration_status.append([self.total_episode_score_so_far,
                                                                   self.episode_actions + [self.end_of_episode_symbol],
                                                                   self.turn_off_exploration])

    def store_episode_in_memory_shaper(self):
        """Stores the raw state, next state, reward, done and action information for the latest full episode"""
        self.memory_shaper.add_episode_experience(self.episode_states, self.episode_next_states, self.episode_rewards,
                                                  self.episode_actions, self.episode_dones)

    def calculate_q_values(self, network_action_values):

        for action_id in range(self.action_size):
            if action_id in self.action_id_to_stepping_stone_action_id.keys():
                stepping_stone_id = self.action_id_to_stepping_stone_action_id[action_id]
                # should do this with no grad? Or grad?
                network_action_values[:, action_id] += network_action_values[:, stepping_stone_id].detach()
        assert network_action_values.shape == (self.hyperparameters["batch_size"], self.action_size)
        return network_action_values

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network. Double DQN
        uses the local index to pick the maximum q_value action and then the target network to calculate the q_value.
        The reasoning behind this is that it will help stop the network from overestimating q values"""
        max_action_indexes = self.calculate_q_values(self.q_network_local(next_states)).detach().argmax(1)
        Q_targets_next = self.calculate_q_values(self.q_network_target(next_states)).gather(1, max_action_indexes.unsqueeze(1))
        return Q_targets_next

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        # must convert actions to long so can be used as index
        Q_expected = self.calculate_q_values(self.q_network_local(states)).gather(1, actions.long())
        return Q_expected

    def abandon_macro_action(self):
        """Need to implement this logic..
        and also decide on the intrinsic rewards when a macro action gets abandoned
        Should there be a punishment?
        """
        return False




    # def play_agent_until_progress_made(self, agent):
    #     """Have the agent play until enough progress is made"""
    #     remaining_episodes_to_play = self.num_episodes - self.episodes_conducted
    #     episode_actions_scores_and_exploration_status = \
    #         agent.run_n_episodes(num_episodes=remaining_episodes_to_play,
    #                              stop_when_progress_made=True,
    #                              episodes_to_run_with_no_exploration=self.episodes_to_run_with_no_exploration)
    #     return episode_actions_scores_and_exploration_status
    #
    # def determine_rolling_score_to_beat_before_recalculating_grammar(self):
    #     """Determines the rollowing window score the agent needs to get to before we will adapt its actions"""
    #     if self.episode_actions_scores_and_exploration_status is not None:
    #         episode_scores = [data[0] for data in self.episode_actions_scores_and_exploration_status]
    #         current_score = np.mean(episode_scores[-self.episodes_to_run_with_no_exploration:])
    #     else:
    #         current_score = self.rolling_score
    #     improvement_required = self.average_score_required_to_win - current_score
    #     target_rolling_score = current_score + (improvement_required * 0.25)
    #     print("NEW TARGET ROLLING SCORE ", target_rolling_score)
    #     return target_rolling_score