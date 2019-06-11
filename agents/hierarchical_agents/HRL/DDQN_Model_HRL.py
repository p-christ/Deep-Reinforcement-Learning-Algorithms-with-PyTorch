import copy
import torch
import random
import numpy as np
import torch.nn.functional as F

from collections import Counter
from torch import optim
from Base_Agent import Base_Agent
from Replay_Buffer import Replay_Buffer
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration


class DDQN_Wrapper(Base_Agent):

    def __init__(self, config, global_action_id_to_primitive_actions, action_length_reward_bonus, end_of_episode_symbol = "/"):
        super().__init__(config)
        self.end_of_episode_symbol = end_of_episode_symbol
        self.global_action_id_to_primitive_actions = global_action_id_to_primitive_actions
        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"], config.seed)
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)

        self.oracle = self.create_oracle()
        self.oracle_optimizer = optim.Adam(self.oracle.parameters(), lr=self.hyperparameters["learning_rate"])

        self.q_network_local = self.create_NN(input_dim=self.state_size + 1, output_dim=self.action_size)
        self.q_network_local.print_model_summary()
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(), lr=self.hyperparameters["learning_rate"])
        self.q_network_target = self.create_NN(input_dim=self.state_size + 1, output_dim=self.action_size)
        Base_Agent.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)

        self.action_length_reward_bonus = action_length_reward_bonus
        self.abandon_ship = config.hyperparameters["abandon_ship"]

    def create_oracle(self):
        """Creates the network we will use to predict the next state"""
        oracle_hyperparameters = copy.deepcopy(self.hyperparameters)
        oracle_hyperparameters["columns_of_data_to_be_embedded"] = []
        oracle_hyperparameters["embedding_dimensions"] = []
        oracle_hyperparameters["linear_hidden_units"] = [5, 5]
        oracle_hyperparameters["final_layer_activation"] = [None, "tanh"]
        oracle = self.create_NN(input_dim=self.state_size + 2, output_dim=[self.state_size + 1, 1], hyperparameters=oracle_hyperparameters)
        oracle.print_model_summary()
        return oracle

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

    def step(self):
        """Runs a step within a game including a learning step if required"""
        step_number = 0.0
        self.state = np.append(self.state, step_number / 200.0) #Divide by 200 because there are 200 steps in cart pole

        self.total_episode_score_so_far = 0
        episode_macro_actions = []
        while not self.done:
            surprised = False
            macro_action = self.pick_action()
            primitive_actions = self.global_action_id_to_primitive_actions[macro_action]
            primitive_actions_conducted = 0
            for ix, action in enumerate(primitive_actions):
                if self.abandon_ship and primitive_actions_conducted > 0:
                    if self.abandon_macro_action(action):
                        break

                step_number += 1
                self.action = action
                self.next_state, self.reward, self.done, _ = self.environment.step(action)
                self.next_state = np.append(self.next_state, step_number  / 200.0) #Divide by 200 because there are 200 steps in cart pole

                self.total_episode_score_so_far += self.reward
                if self.hyperparameters["clip_rewards"]: self.reward = max(min(self.reward, 1.0), -1.0)
                primitive_actions_conducted += 1
                self.track_episodes_data()
                self.save_experience()

                if len(primitive_actions) > 1:

                    surprised = self.am_i_surprised()


                self.state = self.next_state
                if self.time_for_q_network_to_learn():
                    for _ in range(self.hyperparameters["learning_iterations"]):
                        self.q_network_learn()
                        self.oracle_learn()
                if self.done or surprised: break
            episode_macro_actions.append(macro_action)
            self.round_of_macro_actions.append(macro_action)
        if random.random() < 0.1: print(Counter(episode_macro_actions))
        self.save_episode_actions_with_score()
        self.episode_number += 1
        self.logger.info("END OF EPISODE")

    def am_i_surprised(self):
        """Returns boolean indicating whether the next_state was a surprise or not"""
        with torch.no_grad():
            state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
            action = torch.Tensor([[self.action]])


            states_and_actions = torch.cat((state, action), dim=1) #must change this for all games besides cart pole
            predictions = self.oracle(states_and_actions)
            predicted_next_state = predictions[0, :-1]

            difference = F.mse_loss(predicted_next_state, torch.Tensor(self.next_state))
            if difference > 0.5:
                print("Surprise! Loss {} -- {} vs. {}".format(difference, predicted_next_state, self.next_state))
                return True
            else: return False


    def abandon_macro_action(self, action):
        """Returns boolean indicating whether to abandon macro action or not"""
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            primitive_q_values = self.calculate_q_values(state, local=True, primitive_actions_only=True)
        q_value_highest = torch.max(primitive_q_values)
        q_values_action = primitive_q_values[:, action]
        if q_value_highest > 0.0: multiplier = 0.7
        else: multiplier = 1.3
        if q_values_action < multiplier * q_value_highest:
            print("BREAKING Action {} -- Q Values {}".format(action, primitive_q_values))
            return True
        else:
            return False

    def pick_action(self, state=None):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        if state is None: state = self.state
        if isinstance(state, np.int64) or isinstance(state, int): state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)
        self.q_network_local.eval() #puts network in evaluation mode
        with torch.no_grad():
            action_values = self.calculate_q_values(state, local=True, primitive_actions_only=False)
        self.q_network_local.train() #puts network back in training mode
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})
        self.logger.info("Q values {} -- Action chosen {}".format(action_values, action))
        return action

    def calculate_q_values(self, states, local, primitive_actions_only):
        """Calculates the q values using the local q network"""
        if local:
            primitive_q_values = self.q_network_local(states)
        else:
            primitive_q_values = self.q_network_target(states)

        num_actions = len(self.global_action_id_to_primitive_actions)
        if primitive_actions_only or num_actions <= self.action_size:
            return primitive_q_values

        extra_q_values = self.calculate_macro_action_q_values(states, num_actions)
        extra_q_values = torch.Tensor([extra_q_values])
        all_q_values = torch.cat((primitive_q_values, extra_q_values), dim=1)

        return all_q_values

    def calculate_macro_action_q_values(self, state, num_actions):
        assert state.shape[0] == 1
        q_values = []
        for action_id in range(self.action_size, num_actions):
            macro_action = self.global_action_id_to_primitive_actions[action_id]
            predicted_next_state = state
            cumulated_reward = 0
            action_ix = 0
            for action in macro_action[:-1]:
                predictions = self.oracle(torch.cat((predicted_next_state, torch.Tensor([[action]])), dim=1))
                rewards = predictions[:, -1]
                predicted_next_state = predictions[:, :-1]
                cumulated_reward += (rewards.item() + self.action_length_reward_bonus) * self.hyperparameters["discount_rate"] ** (action_ix)
                action_ix += 1
            final_action = macro_action[-1]
            final_q_value = self.q_network_local(predicted_next_state)[0, final_action]
            total_q_value = cumulated_reward + final_q_value * self.hyperparameters["discount_rate"] ** (action_ix)
            q_values.append(total_q_value)
        return q_values

    def time_for_q_network_to_learn(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin and there are
        enough experiences in the replay buffer to learn from"""
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()

    def right_amount_of_steps_taken(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin"""
        return self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def q_network_learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        if experiences is None: states, actions, rewards, next_states, dones = self.sample_experiences() #Sample experiences
        else: states, actions, rewards, next_states, dones = experiences
        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss, self.hyperparameters["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.q_network_local, self.q_network_target,
                                           self.hyperparameters["tau"])

    def sample_experiences(self):
        """Draws a random sample of experience from the memory buffer"""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            max_action_indexes = self.calculate_q_values(next_states, local=True, primitive_actions_only=True).detach().argmax(1)
            Q_targets_next = self.calculate_q_values(next_states, local=False, primitive_actions_only=True).gather(1,max_action_indexes.unsqueeze(1))
            Q_targets = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))
        Q_expected = self.calculate_q_values(states, local=True, primitive_actions_only=True).gather(1,actions.long())  # must convert actions to long so can be used as index
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def save_episode_actions_with_score(self):

        self.episode_actions_scores_and_exploration_status.append([self.total_episode_score_so_far,
                                                                   self.episode_actions + [self.end_of_episode_symbol],
                                                                   self.turn_off_exploration])

    def oracle_learn(self):
        states, actions, rewards, next_states, _ = self.sample_experiences()  # Sample experiences
        states_and_actions = torch.cat((states, actions), dim=1) #must change this for all games besides cart pole
        predictions = self.oracle(states_and_actions)
        loss = F.mse_loss(torch.cat((next_states, rewards), dim=1), predictions) / float(next_states.shape[1] + 1.0)
        self.take_optimisation_step(self.oracle_optimizer, self.oracle,
                                    loss, self.hyperparameters["gradient_clipping_norm"])
        self.logger.info("Oracle Loss {}".format(loss))


    # def create_feature_extractor(self):
    #     """Creates the feature extractor local network and target network. This means that the q_network and oracle
    #     only need 1 layer"""
    #     temp_hyperparameters = copy.deepcopy(self.hyperparameters)
    #     temp_hyperparameters["linear_hidden_units"], output_dim = temp_hyperparameters["linear_hidden_units"][:-1], temp_hyperparameters["linear_hidden_units"][-1]
    #     temp_hyperparameters["final_layer_activation"] = "relu"
    #     feature_extractor_local = self.create_NN(input_dim=self.state_size, output_dim=output_dim, hyperparameters=temp_hyperparameters)
    #     feature_extractor_target = self.create_NN(input_dim=self.state_size, output_dim=output_dim,hyperparameters=temp_hyperparameters)
    #     Base_Agent.copy_model_over(from_model=feature_extractor_local, to_model=feature_extractor_target)
    #     feature_extractor_local.print_model_summary()
    #     return feature_extractor_local, feature_extractor_target, output_dim
