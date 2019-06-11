import random
from collections import Counter

import torch
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

class DDQN_Wrapper(DDQN):

    def __init__(self, config, global_action_id_to_primitive_action, end_of_episode_symbol="/"):
        super().__init__(config)
        self.state_size += 1

        self.q_network_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyperparameters["learning_rate"])
        self.q_network_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        Base_Agent.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)

        self.min_episode_score_seen = float("inf")
        self.end_of_episode_symbol = end_of_episode_symbol
        self.global_action_id_to_primitive_action = global_action_id_to_primitive_action
        self.action_id_to_stepping_stone_action_id = {}
        self.calculate_q_values_as_increments = self.config.hyperparameters["calculate_q_values_as_increments"]
        self.abandon_ship = self.config.hyperparameters["abandon_ship"]
        self.pre_training_learning_iterations_multiplier = self.hyperparameters[
            "pre_training_learning_iterations_multiplier"]
        self.copy_over_hidden_layers = self.hyperparameters["copy_over_hidden_layers"]
        self.action_balanced_replay_buffer = self.hyperparameters["action_balanced_replay_buffer"]
        self.original_primitive_actions = list(range(self.action_size))
        self.memory_shaper = Memory_Shaper(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"], config.seed,
                                           self.update_reward_to_encourage_longer_macro_actions, self.action_balanced_replay_buffer)
        self.action_length_reward_bonus = self.hyperparameters["action_length_reward_bonus"]
        self.only_train_new_actions = self.hyperparameters["only_train_new_actions"]
        self.only_train_final_layer = self.hyperparameters["only_train_final_layer"]


    def update_agent(self, global_action_id_to_primitive_action, new_actions_just_added):
        """Updates the agent according to new action set by changing its action set, creating a new replay buffer
        and doing any pretraining"""
        current_num_actions = len(global_action_id_to_primitive_action.keys())
        PRE_TRAINING_ITERATIONS = int(self.pre_training_learning_iterations_multiplier)
        self.update_agent_for_new_actions(global_action_id_to_primitive_action,
                                                copy_over_hidden_layers=self.copy_over_hidden_layers,
                                                change_or_append_final_layer="APPEND")
        if len(new_actions_just_added) > 0:
            replay_buffer = self.memory_shaper.put_adapted_experiences_in_a_replay_buffer(
                global_action_id_to_primitive_action)
            self.overwrite_replay_buffer_and_pre_train_agent(replay_buffer, PRE_TRAINING_ITERATIONS,
                                                             only_train_final_layer=self.only_train_final_layer,
                                                             only_train_new_actions=self.only_train_new_actions,
                                                             new_actions_just_added=new_actions_just_added)
        print("Now there are {} actions: {}".format(current_num_actions, self.global_action_id_to_primitive_action))


    def overwrite_replay_buffer_and_pre_train_agent(self, replay_buffer, training_iterations, only_train_final_layer,
                                                    only_train_new_actions, new_actions_just_added):
        """Overwrites the replay buffer of the agent and sets it to the provided replay_buffer. Then trains the agent
        for training_iterations number of iterations using data from the replay buffer"""
        assert replay_buffer is not None
        self.memory = replay_buffer
        if only_train_final_layer:
            print("Only training the final layer")
            self.freeze_all_but_output_layers(self.q_network_local)

        for g in self.q_network_optimizer.param_groups:
            g['lr'] = self.hyperparameters["learning_rate"] / 100.0
        for _ in range(training_iterations):
            if only_train_new_actions: new_actions = new_actions_just_added
            else: new_actions = []
            self.learn(print_loss=False, only_these_actions=new_actions)
        for g in self.q_network_optimizer.param_groups:
            g['lr'] = self.hyperparameters["learning_rate"]
        if only_train_final_layer: self.unfreeze_all_layers(self.q_network_local)

    def update_agent_for_new_actions(self, global_action_id_to_primitive_action, copy_over_hidden_layers, change_or_append_final_layer):
        assert change_or_append_final_layer in ["CHANGE", "APPEND"]
        num_actions_before = self.action_size
        self.global_action_id_to_primitive_action = global_action_id_to_primitive_action
        self.action_size = len(global_action_id_to_primitive_action)
        num_new_actions = self.action_size - num_actions_before
        if num_new_actions > 0:
            for new_action_id in range(num_actions_before, num_actions_before + num_new_actions):
                self.update_action_id_to_stepping_stone_action_id(new_action_id)
            if change_or_append_final_layer == "CHANGE": self.change_final_layer_q_network(copy_over_hidden_layers)
            else: self.append_to_final_layers(num_new_actions)

    def update_action_id_to_stepping_stone_action_id(self, new_action_id):
        """Update action_id_to_stepping_stone_action_id with the new actions created"""
        new_action = self.global_action_id_to_primitive_action[new_action_id]
        length_macro_action = len(new_action)
        print(" update_action_id_to_stepping_stone_action_id ")
        for sub_action_length in reversed(range(1, length_macro_action)):
            sub_action = new_action[:sub_action_length]
            if sub_action in self.global_action_id_to_primitive_action.values():
                sub_action_id = list(self.global_action_id_to_primitive_action.keys())[
                    list(self.global_action_id_to_primitive_action.values()).index(sub_action)]

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
        if len(only_these_actions) == 0:
            super().learn()
            # self.learn_predict_next_state()
        else:
            experiences = self.memory.sample_experiences_with_certain_actions(only_these_actions, self.action_size,
                                                                              int(self.hyperparameters["batch_size"]))
            super().learn(experiences=experiences)

    def update_reward_to_encourage_longer_macro_actions(self, cumulative_reward, length_of_macro_action):
        """Update reward to encourage usage of longer macro actions. The size of the improvement depends positively
        on the length of the macro action"""
        if cumulative_reward == 0.0: increment = 0.1
        else: increment = abs(cumulative_reward)
        total_change = increment * ((length_of_macro_action - 1)** 0.5) * self.action_length_reward_bonus
        cumulative_reward += total_change
        return cumulative_reward

    def step(self):
        """Runs a step within a game including a learning step if required"""
        self.total_episode_score_so_far = 0
        step_number = 0.0
        self.state = np.append(self.state, step_number / 200.0) #Divide by 200 because there are 200 steps in cart pole

        macro_state = self.state
        state = self.state
        done = self.done

        episode_macro_actions = []

        while not done:
            macro_action = self.pick_action(state=macro_state)
            primitive_actions = self.global_action_id_to_primitive_action[macro_action]
            macro_reward = 0
            primitive_actions_conducted = 0
            for action in primitive_actions:


                if self.abandon_ship:

                    if primitive_actions_conducted >= 1:

                        if isinstance(state, np.int64) or isinstance(state, int): state_tensor = np.array([state])
                        else: state_tensor = state
                        state_tensor = torch.from_numpy(state_tensor).float().unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            q_values = self.calculate_q_values(self.q_network_local(state_tensor))[:, :self.get_action_size()]
                        q_value_highest = torch.max(q_values)
                        q_values_action = q_values[:, action]

                        if q_value_highest == 0.0:
                            increment = 1.0
                        else:
                            increment = abs(q_value_highest)


                        max_difference = 0.2 * increment
                        if q_values_action + max_difference < q_value_highest:
                            # print("BREAKING Action {} -- Q Values {}".format(action, q_values))
                            macro_reward -= 0.25  #punish agent for picking macro action that it had to pull out of
                            # break
                            print("Changing Course of Action {} to {} -- Q Values {}".format(action, torch.argmax(q_values), q_values))
                            action = torch.argmax(q_values).item()

                step_number += 1

                next_state, reward, done, _ = self.environment.step(action)
                self.total_episode_score_so_far += reward
                if self.hyperparameters["clip_rewards"]: reward = max(min(reward, 1.0), -1.0)
                macro_reward += reward
                primitive_actions_conducted += 1
                next_state = np.append(next_state, step_number / 200.0) #Divide by 200 because there are 200 steps in cart pole
                self.track_episodes_data(state, action, reward, next_state, done)

                self.save_experience(experience=(state, action, reward, next_state, done))


                state = next_state
                if self.time_for_q_network_to_learn():
                    for _ in range(self.hyperparameters["learning_iterations"]):
                        self.learn()
                if done: break

            macro_reward = self.update_reward_to_encourage_longer_macro_actions(macro_reward, primitive_actions_conducted)
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

        if not self.calculate_q_values_as_increments: return network_action_values

        for action_id in range(self.action_size):
            if action_id in self.action_id_to_stepping_stone_action_id.keys():
                stepping_stone_id = self.action_id_to_stepping_stone_action_id[action_id]
                # should do this with no grad? Or grad?
                with torch.no_grad():
                    network_action_values[:, action_id] += network_action_values[:, stepping_stone_id] #.detach()
        # assert network_action_values.shape[0] in set([self.hyperparameters["batch_size"], 1])
        assert network_action_values.shape[1] == self.action_size
        return network_action_values


    def pick_action(self, state=None):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if state is None: state = self.state
        if isinstance(state, np.int64) or isinstance(state, int): state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)
        self.q_network_local.eval() #puts network in evaluation mode
        with torch.no_grad():
            action_values = self.calculate_q_values(self.q_network_local(state))
        self.q_network_local.train() #puts network back in training mode
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})
        self.logger.info("Q values {} -- Action chosen {}".format(action_values, action))
        return action


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
