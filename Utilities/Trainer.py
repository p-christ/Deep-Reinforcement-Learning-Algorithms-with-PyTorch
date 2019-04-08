import copy
import random
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

class Trainer(object):

    def __init__(self, config, agents):
        self.config = config
        self.agents = agents
        self.agent_to_agent_group = self.create_agent_to_agent_group_dictionary()
        self.results = None
        self.colors = ["red", "blue", "green", "orange", "yellow", "purple"]
        self.colour_ix = 0

    def create_agent_to_agent_group_dictionary(self):
        """Creates a dictionary that maps an agent to their wider agent group"""
        create_agent_to_agent_group_dictionary = {
            "DQN": "DQN_Agents",
            "DQN_HER": "DQN_Agents",
            "DDQN": "DQN_Agents",
            "DDQN with Prioritised Replay": "DQN_Agents",
            "DQN with Fixed Q Targets": "DQN_Agents",
            "Duelling DQN": "DQN_Agents",
            "PPO": "Policy_Gradient_Agents",
            "REINFORCE": "Policy_Gradient_Agents",
            "Genetic_Agent": "Stochastic_Policy_Search_Agents",
            "Hill Climbing": "Stochastic_Policy_Search_Agents",
            "DDPG": "Actor_Critic_Agents",
            "DDPG_HER": "Actor_Critic_Agents"
        }
        return create_agent_to_agent_group_dictionary

    def run_games_for_agents(self):
        """Run a set of games for each agent. Optionally visualising and/or saving the results"""
        self.results = self.create_object_to_store_results()
        for agent_number, agent_class in enumerate(self.agents):
            agent_name = agent_class.agent_name
            self.run_games_for_agent(agent_number + 1, agent_class)
            if self.config.visualise_overall_agent_results:
                agent_rolling_score_results = [results[1] for results in  self.results[agent_name]]
                self.visualise_overall_agent_results(agent_rolling_score_results, agent_name, show_mean_and_std_range=True)

        if self.config.file_to_save_results_graph: plt.savefig(self.config.file_to_save_results_graph, bbox_inches="tight")
        if self.config.file_to_save_data_results: self.save_obj(self.results, self.config.file_to_save_data_results)
        plt.show()
        # if self.config.visualise_overall_results:

    def visualise_overall_agent_results(self, agent_results, agent_name, show_mean_and_std_range=False, show_each_run=False):
        """Visualises the results for one agent"""
        assert isinstance(agent_results, list), "agent_results must be a list of lists, 1 set of results per list"
        assert isinstance(agent_results[0], list), "agent_results must be a list of lists, 1 set of results per list"
        assert bool(show_mean_and_std_range) ^ bool(show_each_run), "either show_mean_and_std_range or show_each_run must be true"

        color = self.get_next_color()
        if show_mean_and_std_range:
            mean_minus_x_std, mean_results, mean_plus_x_std = self.get_mean_and_standard_deviation_difference_results(agent_results)
            # Ignore data after the mean result solves the game
            # mean_minus_x_std, mean_results, mean_plus_x_std = self.ignore_points_after_game_solved(mean_minus_x_std, mean_results, mean_plus_x_std)
            x_vals = list(range(len(mean_results)))
            plt.plot(x_vals, mean_results, label=agent_name, color=color)
            plt.plot(x_vals, mean_plus_x_std, color=color, alpha=0.1)
            plt.plot(x_vals, mean_minus_x_std, color=color, alpha=0.1)
            plt.fill_between(x_vals, y1=mean_minus_x_std, y2=mean_plus_x_std, alpha=0.1, color=color)
        else:
            for ix, result in enumerate(agent_results):
                x_vals = list(range(len(agent_results[0])))
                plt.plot(x_vals, result, label=agent_name + "_{}".format(ix+1), color=color)
                color = self.get_next_color()
        ax = plt.gca()
        ax.set_facecolor('xkcd:white')

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.05,
                         box.width, box.height * 0.95])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fancybox=True, shadow=True, ncol=3)

        plt.title(self.config.environment.environment_name, fontsize=15, fontweight='bold')

        plt.ylabel('Rolling Episode Scores')
        plt.xlabel('Episode Number')

        self.hide_spines(ax, ['right', 'top'])
        ax.set_xlim([0, self.config.num_episodes_to_run])

        if self.config.show_solution_score:
            self.draw_horizontal_line_with_label(ax, y_value=self.config.environment.get_score_to_win(), x_min=0,
                                        x_max=self.config.num_episodes_to_run * 1.02, label="Target \n score")

    def hide_spines(self, ax, spines_to_hide):
        for spine in spines_to_hide:
            ax.spines[spine].set_visible(False)

    def get_mean_and_standard_deviation_difference_results(self, results):
        """From a list of lists of agent results it extracts the mean results and the mean results plus or minus
         some multiple of the standard deviation"""
        def get_results_at_a_time_step(results, timestep):
            results_at_a_time_step = [result[timestep] for result in results]
            return results_at_a_time_step
        def get_standard_deviation_at_time_step(results, timestep):
            results_at_a_time_step = [result[timestep] for result in results]
            return np.std(results_at_a_time_step)
        mean_results = [np.mean(get_results_at_a_time_step(results, timestep)) for timestep in range(len(results[0]))]
        mean_minus_x_std = [mean_val - self.config.standard_deviation_results * get_standard_deviation_at_time_step(results, timestep) for
                            timestep, mean_val in enumerate(mean_results)]
        mean_plus_x_std = [mean_val + self.config.standard_deviation_results * get_standard_deviation_at_time_step(results, timestep) for
                           timestep, mean_val in enumerate(mean_results)]
        return mean_minus_x_std, mean_results, mean_plus_x_std

    def get_next_color(self):
        """Gets the next color in list self.colors. If it gets to the end then it starts from beginning"""
        self.colour_ix += 1
        if self.colour_ix >= len(self.colors): self.colour_ix = 0
        color = self.colors[self.colour_ix]
        return color

    def ignore_points_after_game_solved(self, mean_minus_x_std, mean_results, mean_plus_x_std):

        for ix in range(len(mean_results)):
            if mean_results[ix] >= self.config.environment.get_score_to_win():
                break

        return mean_minus_x_std[:ix], mean_results[:ix], mean_plus_x_std[:ix]

    def draw_horizontal_line_with_label(self, ax, y_value, x_min, x_max, label):
        ax.hlines(y=y_value, xmin=x_min, xmax=x_max,
                  linewidth=2, color='k', linestyles='dotted', alpha=0.5)
        ax.text(x_max, y_value * 0.965, label)

    def run_games_for_agent(self, agent_number, agent_class):
        """Runs a set of games for a given agent, saving the results in self.results"""
        agent_results = []
        agent_name = agent_class.agent_name
        agent_group = self.agent_to_agent_group[agent_name]
        agent_config = copy.deepcopy(self.config)
        agent_config.hyperparameters = self.config.hyperparameters[agent_group]
        agent_round = 1
        for run in range(self.config.runs_per_agent):
            print("AGENT NAME: {}".format(agent_name))
            print("\033[1m" + "{}.{}: {}".format(agent_number, agent_round, agent_name) + "\033[0m", flush=True)
            if self.config.randomise_random_seed: agent_config.seed = random.randint(0, 1000)
            agent = agent_class(agent_config)
            game_scores, rolling_scores, time_taken = agent.run_n_episodes()
            print("Time taken: {}".format(time_taken), flush=True)
            self.print_two_empty_lines()
            agent_results.append([game_scores, rolling_scores, len(rolling_scores), -1 * max(rolling_scores), time_taken])
            if self.config.visualise_individual_results:
                self.visualise_overall_agent_results([rolling_scores], agent_name, show_each_run=True)
                plt.show()
            agent_round += 1
        self.results[agent_name] = agent_results

    def create_object_to_store_results(self):
        """Creates a dictionary that we will store the results in"""
        if self.config.overwrite_existing_results_file or not self.config.file_to_save_data_results or not os.path.isfile(self.config.file_to_save_data_results):
            results = {}
        else: results = self.load_obj(self.config.file_to_save_data_results)
        return results

    def print_two_empty_lines(self):
        print("-----------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------")
        print(" ")

    def save_obj(self, obj, name):
        """Saves given object as a pickle file"""
        if name[-4:] != ".pkl":
            name += ".pkl"
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        """Loads a pickle file object"""
        with open(name, 'rb') as f:
            return pickle.load(f)

    def visualise_preexisting_results(self, save_image_path=None):
        """Visualises saved data results and then optionally saves the image"""
        preexisting_results = self.create_object_to_store_results()
        for agent in preexisting_results.keys():
            agent_rolling_score_results = [results[1] for results in preexisting_results[agent]]
            self.visualise_overall_agent_results(agent_rolling_score_results, agent, show_mean_and_std_range=True)
        if save_image_path: plt.savefig(save_image_path, bbox_inches="tight")
        plt.show()


