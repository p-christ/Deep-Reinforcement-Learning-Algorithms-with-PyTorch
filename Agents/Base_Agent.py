import sys
import gym
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

class Base_Agent(object):
    
    def __init__(self, config):
        self.set_random_seeds(config.seed)
        self.environment = config.environment
        self.action_size = self.environment.get_action_size()
        self.action_types = self.environment.get_action_types()
        self.state_size = self.environment.get_state_size()
        self.hyperparameters = config.hyperparameters
        self.average_score_required_to_win = self.environment.get_score_to_win()
        self.rolling_score_window = self.environment.get_rolling_period_to_calculate_score_over()
        self.max_steps_per_episode = self.environment.get_max_steps_per_episode()
        self.total_episode_score_so_far = 0
        self.game_full_episode_scores = []
        self.rolling_results = []
        self.max_rolling_score_seen = float("-inf")
        self.max_episode_score_seen = float("-inf")
        self.episode_number = 0
        self.device = "cuda:0" if config.use_GPU else "cpu"
        self.visualise_results_boolean = config.visualise_individual_results
        self.run_checks()
        gym.logger.set_level(40)  # stops it from printing an unnecessary warning

    def set_random_seeds(self, seed):
        self.random_seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.environment.reset_environment()
        self.state = self.environment.get_state()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_step_number = 0
        self.episode_states = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []

    def track_episodes_data(self):
        self.episode_states.append(self.state)
        self.episode_actions.append(self.action)
        self.episode_next_states.append(self.next_state)
        self.episode_dones.append(self.done)

    def run_n_episodes(self, num_episodes_to_run=1, save_model=False):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        start = time.time()
        while self.episode_number < num_episodes_to_run:
            self.reset_game()
            self.step()
            self.save_and_print_result()
            if self.max_rolling_score_seen > self.average_score_required_to_win: #stop once we achieve required score
                break
        time_taken = time.time() - start
        self.summarise_results()
        if save_model:
            self.locally_save_policy()
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def step(self):
        raise ValueError("Step needs to be implemented by the agent")

    def conduct_action(self):
        self.environment.conduct_action(self.action)

    def update_next_state_reward_done_and_score(self): 
        self.next_state = self.environment.get_next_state()
        self.reward = self.environment.get_reward()
        self.done = self.environment.get_done()
        self.total_episode_score_so_far += self.environment.get_reward()

    def save_and_print_result(self):
        self.save_result()
        self.print_rolling_result()

    def save_result(self):
        self.game_full_episode_scores.append(self.total_episode_score_so_far)
        self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
        self.save_max_result_seen()

    def save_max_result_seen(self):
        if self.game_full_episode_scores[-1] > self.max_episode_score_seen:
            self.max_episode_score_seen = self.game_full_episode_scores[-1]

        if self.rolling_results[-1] > self.max_rolling_score_seen:
            if len(self.rolling_results) > self.rolling_score_window:
                self.max_rolling_score_seen = self.rolling_results[-1]

    def print_rolling_result(self):
        sys.stdout.write(
            """"\r Episode {0}, Score: {3: .2f}, Max score seen: {4: .2f},  Rolling score: {1: .2f}, Max rolling score seen: {2: .2f}""".format(len(self.game_full_episode_scores),
                                                                                               self.rolling_results[-1],
                                                                                               self.max_rolling_score_seen,
                                                                                               self.game_full_episode_scores[-1], self.max_episode_score_seen))
        sys.stdout.flush()

    def generate_string_to_print(self):
        return

    def summarise_results(self):
        self.show_whether_achieved_goal()
        if self.visualise_results_boolean:
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
            if score > self.average_score_required_to_win:
                return ix
        return -1

    def visualise_results(self):
        plt.plot(self.rolling_results)
        plt.ylabel('Episode score')
        plt.xlabel('Episode number')
        plt.show()

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
        assert self.action_types in ["DISCRETE", "CONTINUOUS"], "Environment needs to provide action types"

    def enough_experiences_to_learn_from(self):
        return len(self.memory) > self.hyperparameters["batch_size"]

    def pick_and_conduct_action(self):
        self.action = self.pick_action()
        self.conduct_action()

    def save_experience(self):
        self.memory.add_experience(self.state, self.action, self.reward, self.next_state, self.done)

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm):
        optimizer.zero_grad() #reset gradients to 0
        loss.backward() #this calculates the gradients
        torch.nn.utils.clip_grad_norm_(network.parameters(), clipping_norm) #clip gradients to help stabilise training
        optimizer.step() #this applies the gradients
    
    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)