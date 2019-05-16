import random

from gym import Wrapper, spaces
from torch import nn, optim

from Base_Agent import Base_Agent
import copy
import time
import numpy as np
from DDQN import DDQN
from Memory_Shaper import Memory_Shaper
from k_Sequitur import k_Sequitur
import numpy as np
from operator import itemgetter



class HRL(Base_Agent):
    agent_name = "HRL"

    #
    # agent learns in world RANDOMLY first...
    # agent makes some sort of progress...
    # then we infer grammar
    # change final layer, freeze earlier layer
    # train with frozen earlier layers
    # unfreeze earlier layers

    # train model at same time to predict next state...  so it can warn us if we see something that is unusual and so control should go back to agent...
    # also need to encourage agent to pick longer actions rather than just 1 action at a time each time

    """ needs to run episodes with no exploration and use that to create macro actions..."""

    # freeze earlier layers of Q network
    # remove and replace final layer of Q network

    # Memory shaper keeps ALL past experience stored. Turns into replay buffer for you with given rules
    # Need to make sure rules

    # Currently... once something becomes an action it can never go back... and we pretend its a primitive action from then on

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

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):

        #TODO learn grammar from most successful episodes not random ones... with or without exploration on?
        #TODO set the score to beat as % lower than best score seen before without exploration
        #TODO do grammar updates less often as we go...  increase % improvement required as we go?
        #TODO do pre-training until loss stops going down quickly?
        #TODO Use SAC-Discrete instead of DQN?
        #TODO learn grammar of a mixture of best performing episodes when had exploration on vs. not
        #TODO stratified sampling so each action from buffer gets seen equal amount of time
        #TODO higher learning rate when just training final layer?

        if num_episodes is None: num_episodes = self.config.num_episodes_to_run

        self.global_action_id_to_primitive_action = {k: tuple([k]) for k in range(self.action_size)}

        replay_buffer = None
        self.num_episodes = num_episodes
        self.episodes_conducted = 0
        self.grammar_induction_iteration = 0

        while self.episodes_conducted < self.num_episodes:

            current_num_actions = len(self.global_action_id_to_primitive_action.keys())
            PRE_TRAINING_ITERATIONS = self.pre_training_learning_iterations_multiplier * (current_num_actions ** 2)
            self.agent = self.generate_agent_with_updated_action_set(copy_over_hidden_layers=True)
            if replay_buffer is not None:
                print(" ------ ")
                print("Length of buffer {} -- Actions {} -- Pre training iterations {}".format(len(replay_buffer),
                                                                                               current_num_actions,
                                                                                               PRE_TRAINING_ITERATIONS))
                print(" ------ ")
                self.overwrite_replay_buffer_and_pre_train_agent(self.agent, replay_buffer, PRE_TRAINING_ITERATIONS,
                                                                 only_train_final_layer=False)


            print("Now there are {} actions: {}".format(current_num_actions, self.global_action_id_to_primitive_action))

            self.episode_actions_scores_and_exploration_status = self.play_agent_until_progress_made(self.agent)
            self.episodes_conducted += len(self.episode_actions_scores_and_exploration_status)
            actions_to_infer_grammar_from = self.pick_actions_to_infer_grammar_from(self.episode_actions_scores_and_exploration_status)

            # print("macro_actions_seen", exploration_free_macro_actions_seen)
            # latest_macro_actions_seen = macro_actions_seen[-2000:]

            self.update_action_choices(actions_to_infer_grammar_from)

            assert len(set(self.global_action_id_to_primitive_action.values())) == len(self.global_action_id_to_primitive_action.values()), \
            "Not all actions are unique anymore: {}".format(self.global_action_id_to_primitive_action)


            for key, value in self.global_action_id_to_primitive_action.items():
                assert max(value) < self.action_size, "Actions should be in terms of primitive actions"
            replay_buffer = self.memory_shaper.put_adapted_experiences_in_a_replay_buffer(self.global_action_id_to_primitive_action)

            self.grammar_induction_iteration += 1

        return self.game_full_episode_scores, self.rolling_results

    def pick_actions_to_infer_grammar_from(self, episode_actions_scores_and_exploration_status):
        """Takes in data summarising the results of the latest games the agent played and then picks the actions from which
        we want to base the subsequent action grammar on"""

        episode_scores = [data[0] for data in episode_actions_scores_and_exploration_status]
        episode_actions = [data[1] for data in episode_actions_scores_and_exploration_status]

        reverse_ordering = np.argsort(episode_scores)
        top_results = list(reverse_ordering[-self.episodes_to_run_with_no_exploration:])


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

        if cutoff == 0.0:
            print(new_count_symbol)
            assert 1 == 0

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

        new_actions = self.flatten_action_id_to_actions(new_unflattened_actions)

        return new_actions

    def play_agent_until_progress_made(self, agent):
        """Have the agent play until enough progress is made"""
        remaining_episodes_to_play = self.num_episodes - self.episodes_conducted
        episode_actions_scores_and_exploration_status = agent.run_n_episodes(num_episodes=remaining_episodes_to_play,
                                                                 stop_when_progress_made=True, episodes_to_run_with_no_exploration=self.episodes_to_run_with_no_exploration)
        return episode_actions_scores_and_exploration_status

    def generate_agent_with_updated_action_set(self, copy_over_hidden_layers):
        """Generates an agent that acts according to the new updated action set"""
        if copy_over_hidden_layers and self.grammar_induction_iteration > 0:
            saved_network = copy.deepcopy(self.agent.q_network_local)
        updated_config = copy.deepcopy(self.config)
        print("Creating new agent with actions ", self.global_action_id_to_primitive_action)
        updated_config.overwrite_action_size = len(self.global_action_id_to_primitive_action)
        target_rolling_score = self.determine_rolling_score_to_beat_before_recalculating_grammar()
        agent = DDQN_Wrapper(updated_config, target_rolling_score, self.global_action_id_to_primitive_action,
                             self.update_reward_to_encourage_longer_macro_actions, self.memory_shaper)

        if copy_over_hidden_layers and self.grammar_induction_iteration > 0:
            agent = self.copy_over_saved_network_besides_final_layer(saved_network, agent)

        return agent

    def copy_over_saved_network_besides_final_layer(self, saved_network, new_agent):
        """Copies over the hidden layers of a saved network to a new agent (but not the output layer"""
        assert len(new_agent.q_network_local.output_layers) == 1
        output_layer = copy.deepcopy(new_agent.q_network_local.output_layers[0])
        new_agent.q_network_local = saved_network
        new_agent.q_network_local.output_layers[0] = output_layer

        Base_Agent.copy_model_over(from_model=new_agent.q_network_local, to_model=new_agent.q_network_target)
        new_agent.q_network_optimizer = optim.Adam(new_agent.q_network_local.parameters(),
                                              lr=new_agent.hyperparameters["learning_rate"])
        return new_agent

    def freeze_all_but_output_layers(self, network):
        """Freezes all layers except the output layer of a network"""
        print("Freezing hidden layers")
        for param in network.named_parameters():
            param_name = param[0]
            assert "hidden" in param_name or "output" in param_name or "embedding" in param_name, "Name {} of network layers not understood".format(param_name)
            if "output" not in param_name:
                param[1].requires_grad = False

    def unfreeze_all_layers(self, network):
        """Unfreezes all layers of a network"""
        print("Unfreezing all layers")
        for param in network.parameters():
            param.requires_grad = True

    def determine_rolling_score_to_beat_before_recalculating_grammar(self):
        """Determines the rollowing window score the agent needs to get to before we will adapt its actions"""
        if self.episode_actions_scores_and_exploration_status is not None:
            episode_scores = [data[0] for data in self.episode_actions_scores_and_exploration_status]
            current_score = np.mean(episode_scores[-self.episodes_to_run_with_no_exploration:])
        else:
            current_score = self.rolling_score
        improvement_required = self.average_score_required_to_win - current_score
        target_rolling_score = current_score + (improvement_required * 0.25)
        print("NEW TARGET ROLLING SCORE ", target_rolling_score)
        return target_rolling_score


    def overwrite_replay_buffer_and_pre_train_agent(self, agent, replay_buffer, training_iterations, only_train_final_layer):
        """Overwrites the replay buffer of the agent and sets it to the provided replay_buffer. Then trains the agent
        for training_iterations number of iterations using data from the replay buffer"""
        assert replay_buffer is not None
        agent.memory = replay_buffer

        if only_train_final_layer:
            print("Only training the final layer")
            self.freeze_all_but_output_layers(agent.q_network_local)
        for _ in range(training_iterations):
            agent.learn()
        if only_train_final_layer: self.unfreeze_all_layers(agent.q_network_local)

    def flatten_action_id_to_actions(self, action_id_to_actions):
        """Converts the values in an action_id_to_actions dictionary back to the primitive actions they represent"""
        flattened_action_id_to_actions = {}
        for key in action_id_to_actions.keys():
            actions = action_id_to_actions[key]
            raw_actions = self.backtrack_action_to_primitive_actions(actions)
            flattened_action_id_to_actions[key] = raw_actions
        return flattened_action_id_to_actions

    def backtrack_action_to_primitive_actions(self, action_tuple):
        """Converts an action tuple back to the primitive actions it represents in a recursive way."""
        print("Recursing to backtrack on ", action_tuple)
        primitive_actions = range(self.action_size)
        if all(action in primitive_actions for action in action_tuple): return action_tuple #base case
        new_action_tuple = []
        for action in action_tuple:
            if action in primitive_actions: new_action_tuple.append(action)
            else:
                converted_action = self.global_action_id_to_primitive_action[action]
                print(new_action_tuple)
                new_action_tuple.extend(converted_action)
                print("Should have changed: ", new_action_tuple)
        new_action_tuple = tuple(new_action_tuple)
        return self.backtrack_action_to_primitive_actions(new_action_tuple)

    def update_reward_to_encourage_longer_macro_actions(self, cumulative_reward, length_of_macro_action):
        """Update reward to encourage usage of longer macro actions. The size of the improvement depends positively
        on the length of the macro action"""
        if cumulative_reward == 0.0: increment = 0.1
        else: increment = abs(cumulative_reward)
        cumulative_reward += increment * ((length_of_macro_action - 1)** 0.5) * self.action_length_reward_bonus
        return cumulative_reward

class DDQN_Wrapper(DDQN):

    def __init__(self, config, rolling_score_to_beat, action_id_to_primitive_actions, bonus_reward_function,
                 memory_shaper, end_of_episode_symbol="/"):
        super().__init__(config)
        self.min_episode_score_seen = float("inf")
        self.end_of_episode_symbol = end_of_episode_symbol
        self.rolling_score_to_beat = rolling_score_to_beat
        self.enough_progress_made = False
        self.episode_actions_scores_and_exploration_status = []
        self.action_id_to_primitive_actions = action_id_to_primitive_actions
        self.bonus_reward_function = bonus_reward_function
        self.memory_shaper = memory_shaper

    def run_n_episodes(self, num_episodes, episodes_to_run_with_no_exploration, stop_when_progress_made=False):

        episodes_with_no_exploration = 0
        while self.episode_number < num_episodes:
            self.reset_game()
            self.step()
            self.save_and_print_result()
            if self.enough_progress_made:
                episodes_with_no_exploration += 1
            if episodes_with_no_exploration >= episodes_to_run_with_no_exploration:
                break
            if stop_when_progress_made:
                if self.progress_has_been_made() and not self.enough_progress_made:
                    self.enough_progress_made = True
                    self.turn_off_any_epsilon_greedy_exploration()

        assert len(self.episode_actions_scores_and_exploration_status) == self.episode_number
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
        self.episode_actions_scores_and_exploration_status.append([self.total_episode_score_so_far, self.episode_actions + [self.end_of_episode_symbol], self.turn_off_exploration])

    def store_episode_in_memory_shaper(self):
        """Stores the raw state, next state, reward, done and action information for the latest full episode"""
        self.memory_shaper.add_episode_experience(self.episode_states, self.episode_next_states, self.episode_rewards,
                                                  self.episode_actions, self.episode_dones)

    def progress_has_been_made(self):
        """Determines whether enough 'progress' has been made for us to do another round of grammar induction"""
        assert self.average_score_required_to_win < 100000, self.average_score_required_to_win
        if len(self.rolling_results) < 20: return False
        return self.rolling_results[-1] >= self.rolling_score_to_beat

    def abandon_macro_action(self):
        """Need to implement this logic..
        and also decide on the intrinsic rewards when a macro action gets abandoned
        Should there be a punishment?
        """
        return False
