import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta
import operator
import pickle
import os


def run_games_for_agents(config, agents):


    runs_per_agent = config.runs_per_agent
    hyperparameters = config.hyperparameters
    requirements_to_solve_game = config.requirements_to_solve_game
    max_episodes_to_run = config.max_episodes_to_run
    visualise_overall_results = config.visualise_overall_results

    file_to_save_data_results = config.file_to_save_data_results
    file_to_save_data_results_graph = config.file_to_save_data_results_graph



    agent_number = 1
    if os.path.isfile(file_to_save_data_results):
        results = load_obj(file_to_save_data_results)
    else:
        results = {}

    for agent_class in agents:
        agent_results = []
        agent_round = 1
        for run in range(runs_per_agent):
            agent_name = agent_class.__name__
            print("\033[1m" + "{}.{}: {}".format(agent_number, agent_round, agent_name) + "\033[0m", flush=True)
            agent = agent_class(config, hyperparameters, agent_name)
            game_scores, rolling_scores, time_taken = agent.run_n_episodes(num_episodes_to_run=max_episodes_to_run, save_model=False)
            print("Time taken: {}".format(time_taken), flush=True)
            print_two_empty_lines()
            agent_results.append([game_scores, rolling_scores, len(rolling_scores), -1 * max(rolling_scores), time_taken])
            agent_round += 1
        agent_number += 1

        median_result = produce_median_results(agent_results)

        print(median_result)

        results[agent_name] = [median_result[0], median_result[1], median_result[4]]

    if file_to_save_data_results is not None:
        save_obj(results, file_to_save_data_results)

    if visualise_overall_results:
        visualise_results_by_agent(results, requirements_to_solve_game["average_score_required"], file_to_save_data_results_graph)




def save_obj(obj, name):
    if name[-4:] != ".pkl":
        name += ".pkl"
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def produce_median_results(agent_results):

    agent_results = sorted(agent_results, key=operator.itemgetter(2, 3))
    median = agent_results[int(len(agent_results) / 2)]

    return median

def abstract(cls):
    return ABCMeta(cls.__name__, cls.__bases__, dict(cls.__dict__))

def print_two_empty_lines():
    print("-----------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------") 
    print(" ")
    
    
def save_score_results(file_path, results):
    np.save(file_path, results) 
    
    
def visualise_results_by_agent(results, target_score, file_to_save_results_graph):


    agents = results.keys()

    fig, axes = plt.subplots(1, 2, sharex=False, figsize = (14, 9)) # plt.subplots()

    lines = []

    for agent_name in agents:

        rolling_scores = results[agent_name][1]

        lines.append(rolling_scores)

        episodes_seen = len(rolling_scores)

        time_taken = results[agent_name][2]
        starting_point = time_taken / len(rolling_scores)
        time_axes = [starting_point * (t + 1.0) for t in range(len(rolling_scores))]

        axes[0].plot(range(episodes_seen), rolling_scores)
        axes[1].plot(time_axes, rolling_scores)


    max_episodes_seen_by_any_agent = max([len(rolling_scores) for rolling_scores in [results[agent_name][1] for agent_name in agents]])
    max_time_taken_by_any_agent =  max([results[agent_name][2] for agent_name in agents])

    min_score_achieved_by_any_agent = min([min(rolling_scores) for rolling_scores in [results[agent_name][1] for agent_name in agents]])

    draw_horizontal_line_with_label(axes[0], y_value=target_score, x_min=0, x_max=max_episodes_seen_by_any_agent, label="Target \n score")
    draw_horizontal_line_with_label(axes[1], y_value=target_score, x_min=0, x_max=max_time_taken_by_any_agent, label="Target \n score")

    hide_spines(axes[0], ['right', 'top'])
    hide_spines(axes[1], ['right', 'top'])

    set_graph_axis_limits(axes[0], 0, max_episodes_seen_by_any_agent, min_score_achieved_by_any_agent, target_score)
    set_graph_axis_limits(axes[1], 0, max_time_taken_by_any_agent, min_score_achieved_by_any_agent, target_score)


    plt.figlegend(lines, labels=agents, loc='lower center', ncol=3, labelspacing=0.)

    axes[0].set_title("Score vs. Episodes Played", y=1.03, fontweight='bold')
    axes[1].set_title("Score vs. Time Elapsed", y=1.03, fontweight='bold')

    set_graph_labels(axes[0], xlabel='Rolling Episode Scores', ylabel='Episode number')
    set_graph_labels(axes[1], xlabel='Rolling Episode Scores', ylabel='Time in Seconds')

    if file_to_save_results_graph is not None:
        plt.savefig(file_to_save_results_graph, bbox_inches = "tight")
    plt.show()

def draw_horizontal_line_with_label(ax, y_value, x_min, x_max, label):
    ax.hlines(y=y_value, xmin=x_min, xmax=x_max,
              linewidth=2, color='k', linestyles='dotted', alpha=0.5)
    ax.text(x_max, y_value * 0.965, label)

def hide_spines(ax, spines_to_hide):
    for spine in spines_to_hide:
        ax.spines[spine].set_visible(False)


def set_graph_axis_limits(ax, xmin, xmax, ymin, ymax):
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

def set_graph_labels(ax, xlabel, ylabel):
    ax.set_ylabel(xlabel)
    ax.set_xlabel(ylabel)
# #
# results = load_obj("/Users/petroschristodoulou/Documents/Deep_RL_Implementations/Results/Cart_Pole/Results_Data.pkl")
# visualise_results_by_agent(results, 195.0, "hello.png")