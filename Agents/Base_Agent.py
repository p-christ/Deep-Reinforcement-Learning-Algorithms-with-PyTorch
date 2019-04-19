import sys
import gym
import random
import numpy as np
import torch
import time
from nn_builder.pytorch.NN import NN

class Base_Agent(object):
    
    def __init__(self, config):
        self.config = config
        self.set_random_seeds(config.seed)
        self.environment = config.environment
        self.action_size = self.get_action_or_state_size_into_correct_shape(self.environment.action_space.shape)
        self.action_types =  "DISCRETE" if self.environment.action_space.dtype == int  else "CONTINUOUS"
        self.state_size =  self.get_action_or_state_size_into_correct_shape(self.environment.observation_space.shape)
        self.hyperparameters = config.hyperparameters
        self.average_score_required_to_win = self.environment.spec.reward_threshold
        self.rolling_score_window = self.environment.spec.trials
        self.max_steps_per_episode = self.environment.spec.max_episode_steps
        self.total_episode_score_so_far = 0
        self.game_full_episode_scores = []
        self.rolling_results = []
        self.max_rolling_score_seen = float("-inf")
        self.max_episode_score_seen = float("-inf")
        self.episode_number = 0
        self.device = "cuda:0" if config.use_GPU else "cpu"
        self.visualise_results_boolean = config.visualise_individual_results
        self.run_checks()
        self.global_step_number = 0
        gym.logger.set_level(40)  # stops it from printing an unnecessary warning

    @staticmethod
    def get_action_or_state_size_into_correct_shape(size):
        """Gets the action_size or state_size from a gym env into the correct shape for a neural network"""
        if len(size) == 1:
            return size[0]

    def set_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
        self.config.seed = random_seed

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []

    def track_episodes_data(self):
        """Saves the data from the recent episodes"""
        self.episode_states.append(self.state)
        self.episode_actions.append(self.action)
        self.episode_next_states.append(self.next_state)
        self.episode_dones.append(self.done)

    def run_n_episodes(self):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        start = time.time()
        while self.episode_number < self.config.num_episodes_to_run:
            self.reset_game()
            self.step()
            self.save_and_print_result()
        time_taken = time.time() - start
        self.show_whether_achieved_goal()
        if self.config.save_model: self.locally_save_policy()
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def step(self):
        """Takes a step in the game. This method must be overriden by any agent"""
        raise ValueError("Step needs to be implemented by the agent")

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        self.next_state, self.reward, self.done, _ = self.environment.step(action)
        self.total_episode_score_so_far += self.reward

    def save_and_print_result(self):
        """Saves and prints results of the game"""
        self.save_result()
        self.print_rolling_result()

    def save_result(self):
        """Saves the result of an episode of the game"""
        self.game_full_episode_scores.append(self.total_episode_score_so_far)
        self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
        self.save_max_result_seen()

    def save_max_result_seen(self):
        """Updates the best episode result seen so far"""
        if self.game_full_episode_scores[-1] > self.max_episode_score_seen:
            self.max_episode_score_seen = self.game_full_episode_scores[-1]

        if self.rolling_results[-1] > self.max_rolling_score_seen:
            if len(self.rolling_results) > self.rolling_score_window:
                self.max_rolling_score_seen = self.rolling_results[-1]

    def print_rolling_result(self):
        """Prints out the latest episode results"""
        sys.stdout.write(
            """"\r Episode {0}, Score: {3: .2f}, Max score seen: {4: .2f},  Rolling score: {1: .2f}, Max rolling score seen: {2: .2f}""".format(len(self.game_full_episode_scores),
                                                                                               self.rolling_results[-1],
                                                                                               self.max_rolling_score_seen,
                                                                                               self.game_full_episode_scores[-1], self.max_episode_score_seen))
        sys.stdout.flush()

    def show_whether_achieved_goal(self):
        """Prints out whether the agent achieved the environment target goal"""
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
            if score > self.average_score_required_to_win:
                return ix
        return -1

    def update_learning_rate(self, starting_lr,  optimizer):
        """Lowers the learning rate according to how close we are to the solution"""
        if len(self.rolling_results) > 0:
            last_rolling_score = self.rolling_results[-1]
            if last_rolling_score > 0.75 * self.average_score_required_to_win:
                new_lr = starting_lr / 100.0
            elif last_rolling_score > 0.6 * self.average_score_required_to_win:
                new_lr = starting_lr / 20.0
            elif last_rolling_score > 0.5 * self.average_score_required_to_win:
                new_lr = starting_lr / 10.0
            elif last_rolling_score > 0.25 * self.average_score_required_to_win:
                new_lr = starting_lr / 2.0
            else:
                new_lr = starting_lr
            for g in optimizer.param_groups:
                g['lr'] = new_lr

    def run_checks(self):
        """Makes sure the environment action_types are valid"""
        assert self.action_types in ["DISCRETE", "CONTINUOUS"], "Environment needs to provide action types"

    def enough_experiences_to_learn_from(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        return len(self.memory) > self.hyperparameters["batch_size"]

    def pick_and_conduct_action(self):
        """Picks and conducts an action"""
        raise ValueError("CHANGE ME")
        self.action = self.pick_action()
        self.conduct_action()

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None: memory = self.memory
        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm, gradients_given=None):
        optimizer.zero_grad() #reset gradients to 0

        if gradients_given is None:
            loss.backward() #this calculates the gradients
        else:
            for ix, parameters in enumerate(network.parameters()):
                for episode in range(len(gradients_given)):
                    if episode == 0:
                        parameters.grad = gradients_given[episode][ix]
                    else:
                        parameters.grad += gradients_given[episode][ix]

        torch.nn.utils.clip_grad_norm_(network.parameters(), clipping_norm) #clip gradients to help stabilise training
        optimizer.step() #this applies the gradients
    
    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def create_NN(self, input_dim, output_dim, key_to_use=None, override_seed=None):
        """Creates a neural network for the agents to use"""
        if key_to_use:
            hyperparameters = self.hyperparameters[key_to_use]
        else:
            hyperparameters = self.hyperparameters
        if override_seed:
            seed = override_seed
        else:
            seed = self.config.seed

        return NN(input_dim=input_dim, linear_hidden_units=hyperparameters["linear_hidden_units"],
                  output_dim=output_dim, output_activation=hyperparameters["final_layer_activation"],
                  batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
                  hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
                  columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
                  embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"],
                  random_seed=seed).to(self.device)

    def get_updated_epsilon_exploration(self, epsilon=1.0, epsilon_decay_denominator=None):
        """Gets the probability that we just pick a random action. This probability decays the more episodes we have seen"""
        if epsilon_decay_denominator is None: epsilon_decay_denominator = self.hyperparameters["epsilon_decay_rate_denominator"]
        epsilon = epsilon / (1.0 + (self.episode_number / epsilon_decay_denominator))
        return epsilon

    def make_epsilon_greedy_choice(self, action_values, epsilon_decay_denominator=None):
        """Chooses action with highest q_value with probability 1 - epsilon, otherwise picks randomly"""
        epsilon = self.get_updated_epsilon_exploration(epsilon_decay_denominator=epsilon_decay_denominator)
        if random.random() > epsilon: return torch.argmax(action_values).item()
        return random.randint(0, action_values.shape[1] - 1)



