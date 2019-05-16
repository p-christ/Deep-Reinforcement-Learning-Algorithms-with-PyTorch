import random
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

# TODO learn grammar from most successful episodes not random ones... with or without exploration on?
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
# TODO fix the fact that the end of episode symbol is appearing in rules... from this line _, _, macro_action_sequence_appearance_count = grammar_calculator.generate_action_grammar(latest_macro_actions_seen)
# TODO change so its the action rules used in most of best performing episodes that get used rather than
#      those that occur the most overall. because a better episode ends faster and so less occurances of actions!!
#      maybe even pick out the fixed X many actions from the top performing episodes no matter how many episodes

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

        self.global_action_id_to_primitive_action = {k: tuple([k]) for k in range(self.action_size)}

        self.agent = DDQN_Wrapper(config, self.global_action_id_to_primitive_action,
                             self.update_reward_to_encourage_longer_macro_actions, self.memory_shaper)

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):

        if num_episodes is None: num_episodes = self.config.num_episodes_to_run
        self.num_episodes = num_episodes
        self.episodes_conducted = 0
        self.grammar_induction_iteration = 1

        while self.episodes_conducted < self.num_episodes:

            self.episode_actions_scores_and_exploration_status = \
                self.agent.run_n_episodes(num_episodes=self.calculate_how_many_episodes_to_play(),
                                          episodes_to_run_with_no_exploration=self.episodes_to_run_with_no_exploration)

            self.episodes_conducted += len(self.episode_actions_scores_and_exploration_status)
            actions_to_infer_grammar_from = self.pick_actions_to_infer_grammar_from(
                self.episode_actions_scores_and_exploration_status)

            self.update_action_choices(actions_to_infer_grammar_from)

            print("New actions ", self.global_action_id_to_primitive_action)

            assert len(set(self.global_action_id_to_primitive_action.values())) == len(
                self.global_action_id_to_primitive_action.values()), \
                "Not all actions are unique anymore: {}".format(self.global_action_id_to_primitive_action)

            for key, value in self.global_action_id_to_primitive_action.items():
                assert max(value) < self.action_size, "Actions should be in terms of primitive actions"
            replay_buffer = self.memory_shaper.put_adapted_experiences_in_a_replay_buffer(
                self.global_action_id_to_primitive_action)

            self.grammar_induction_iteration += 1

            current_num_actions = len(self.global_action_id_to_primitive_action.keys())
            PRE_TRAINING_ITERATIONS = self.pre_training_learning_iterations_multiplier * (current_num_actions ** 2)

            self.agent.update_agent_for_new_actions(self.global_action_id_to_primitive_action,
                                                    copy_over_hidden_layers=self.copy_over_hidden_layers)


            print(" ------ ")
            print("Length of buffer {} -- Actions {} -- Pre training iterations {}".format(len(replay_buffer),
                                                                                           current_num_actions,
                                                                                           PRE_TRAINING_ITERATIONS))
            print(" ------ ")
            self.overwrite_replay_buffer_and_pre_train_agent(replay_buffer, PRE_TRAINING_ITERATIONS,
                                                             only_train_final_layer=False)
            print("Now there are {} actions: {}".format(current_num_actions, self.global_action_id_to_primitive_action))

        return self.game_full_episode_scores, self.rolling_results

    def calculate_how_many_episodes_to_play(self):
        """Calculates how many episodes the agent should play until we re-infer the grammar"""
        episodes_to_play = self.hyperparameters["epsilon_decay_rate_denominator"] / self.grammar_induction_iteration
        episodes_to_play = int(max(self.episodes_to_run_with_no_exploration * 2, episodes_to_play))
        print("Grammar iteration {} -- Episodes to play {}".format(self.grammar_induction_iteration, episodes_to_play))
        return episodes_to_play

    def pick_actions_to_infer_grammar_from(self, episode_actions_scores_and_exploration_status):
        """Takes in data summarising the results of the latest games the agent played and then picks the actions from which
        we want to base the subsequent action grammar on"""
        episode_scores = [data[0] for data in episode_actions_scores_and_exploration_status]
        episode_actions = [data[1] for data in episode_actions_scores_and_exploration_status]
        reverse_ordering = np.argsort(episode_scores)
        top_results = list(reverse_ordering[-self.episodes_to_run_with_no_exploration*2:])



        best_episode_actions = list(itemgetter(*top_results)(episode_actions))
        best_episode_actions = [item for sublist in best_episode_actions for item in sublist]
        print("Best episode results ", itemgetter(*top_results)(episode_scores))
        return best_episode_actions

    def update_action_choices(self, latest_macro_actions_seen):
        """Creates a grammar out of the latest list of macro actions conducted by the agent"""
        grammar_calculator = k_Sequitur(k=self.config.hyperparameters["sequitur_k"],
                                        end_of_episode_symbol=self.end_of_episode_symbol)
        print("latest_macro_actions_seen ", latest_macro_actions_seen)
        _, _, macro_action_sequence_appearance_count = grammar_calculator.generate_action_grammar(latest_macro_actions_seen)
        print("NEW sequence_appearance_count ", macro_action_sequence_appearance_count)
        new_actions = self.pick_new_macro_actions(macro_action_sequence_appearance_count)
        self.update_global_action_id_to_primitive_action(new_actions)

    def update_global_action_id_to_primitive_action(self, new_actions):
        """Updates global_action_id_to_primitive_action by adding any new actions in that aren't already represented"""
        unique_new_actions = {k: v for k, v in new_actions.items() if v not in self.global_action_id_to_primitive_action.values()}
        next_action_name = max(self.global_action_id_to_primitive_action.keys()) + 1
        for _, value in unique_new_actions.items():
            self.global_action_id_to_primitive_action[next_action_name] = value
            next_action_name += 1

    def pick_new_macro_actions(self, new_count_symbol):
        """Picks the new macro actions to be made available to the agent. Returns them in the form {action_id: (action_1, action_2, ...)}.
        NOTE there are many ways to do this... i should do experiments testing different ways and report the results
        """
        new_unflattened_actions = {}
        total_actions = np.sum([new_count_symbol[rule] for rule in new_count_symbol.keys()])
        cutoff = total_actions * 0.07
        assert cutoff != 0.0, new_count_symbol
        print(" ")
        print("Cutoff ", cutoff)
        print(" ")
        action_id = len(self.global_action_id_to_primitive_action.keys())
        for rule in new_count_symbol.keys():
            count = new_count_symbol[rule]
            print("Rule {} -- Count {}".format(rule, count))
            if count >= cutoff:
                new_unflattened_actions[action_id] = rule
                action_id += 1
        new_actions = flatten_action_id_to_actions(new_unflattened_actions, self.global_action_id_to_primitive_action,
                                                   self.action_size)
        return new_actions

    def overwrite_replay_buffer_and_pre_train_agent(self, replay_buffer, training_iterations, only_train_final_layer):
        """Overwrites the replay buffer of the agent and sets it to the provided replay_buffer. Then trains the agent
        for training_iterations number of iterations using data from the replay buffer"""
        assert replay_buffer is not None
        self.agent.memory = replay_buffer
        if only_train_final_layer:
            print("Only training the final layer")
            self.freeze_all_but_output_layers(self.agent.q_network_local)
        for _ in range(training_iterations):
            self.agent.learn()
        if only_train_final_layer: self.unfreeze_all_layers(self.agent.q_network_local)

    def update_reward_to_encourage_longer_macro_actions(self, cumulative_reward, length_of_macro_action):
        """Update reward to encourage usage of longer macro actions. The size of the improvement depends positively
        on the length of the macro action"""
        if cumulative_reward == 0.0: increment = 0.1
        else: increment = abs(cumulative_reward)
        cumulative_reward += increment * ((length_of_macro_action - 1)** 0.5) * self.action_length_reward_bonus
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

    def update_agent_for_new_actions(self, action_id_to_primitive_actions, copy_over_hidden_layers):
        self.action_id_to_primitive_actions = action_id_to_primitive_actions
        self.action_size = len(action_id_to_primitive_actions)
        self.change_final_layer_q_network(copy_over_hidden_layers)


    def change_final_layer_q_network(self, copy_over_hidden_layers):
        """Changes the final layer of the q network to accomodate the new action space"""
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

        return self.episode_actions_scores_and_exploration_status

    def step(self):
        """Runs a step within a game including a learning step if required"""
        self.total_episode_score_so_far = 0
        macro_state = self.state
        state = self.state
        done = self.done

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
                state = next_state
                if self.time_for_q_network_to_learn():
                    self.learn()
                if done or self.abandon_macro_action(): break

            macro_reward = self.bonus_reward_function(macro_reward, primitive_actions_conducted)
            macro_next_state = next_state
            macro_done = done
            self.save_experience(experience=(macro_state, macro_action, macro_reward, macro_next_state, macro_done))
            macro_state = macro_next_state

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