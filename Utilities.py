import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta
import operator
import pickle
import time
import os


def run_games_for_agents(environment, agents, runs_per_agent, hyperparameters, requirements_to_solve_game,
                         max_episodes_to_run, visualise_results, save_data_filename=None,
                         file_to_save_results_graph=None, seed=100):

    agent_number = 1
    if os.path.isfile(save_data_filename):
        results = load_obj(save_data_filename)
    else:
        results = {}

    for agent_class in agents:
        agent_results = []
        agent_round = 1
        for run in range(runs_per_agent):
            start = time.time()
            agent_name = agent_class.__name__
            print("\033[1m" + "{}.{}: {}".format(agent_number, agent_round, agent_name) + "\033[0m", flush=True)
            agent = agent_class(environment, seed, hyperparameters,
                                requirements_to_solve_game["rolling_score_window"],
                                requirements_to_solve_game["average_score_required"], agent_name)
            game_scores, rolling_scores = agent.run_n_episodes(num_episodes_to_run=max_episodes_to_run, save_model=False)
            print("Time taken: {}".format(time.time() - start), flush=True)
            print_two_empty_lines()
            agent_results.append([game_scores, rolling_scores, len(rolling_scores), -1 * max(rolling_scores)])
            agent_round += 1
        agent_number += 1

        median_result = produce_median_results(agent_results)

        results[agent_name] = [median_result[0], median_result[1]]

    if save_data_filename is not None:
        save_obj(results, save_data_filename)

    if visualise_results:
        visualise_results_by_agent(results, requirements_to_solve_game["average_score_required"], file_to_save_results_graph)




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

    legend_values = []
    max_episodes_seen = 0
    min_score_seen = float("inf")
    fig, ax = plt.subplots()

    for agent_name in agents:

        rolling_scores = results[agent_name][1]

        episodes_seen = len(rolling_scores)

        if episodes_seen > max_episodes_seen:
            max_episodes_seen = episodes_seen

        min_score = min(rolling_scores)
        if min_score < min_score_seen:
            min_score_seen = min_score

        ax.plot(range(episodes_seen), rolling_scores)

        legend_values.append(agent_name)

    # ax.set_position([0.1, 0.1, 0.5, 0.8])

    ax.hlines(y=target_score, xmin=0, xmax=max_episodes_seen,
              linewidth=2, color='k', linestyles='dotted')

    ax.text(max_episodes_seen, target_score * 0.965, "Target \n score")

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.xlim(xmin=0)
    plt.ylim(ymin=min_score_seen)
    plt.ylim(ymax=target_score)

    plt.legend(legend_values, loc='center', bbox_to_anchor=(0.5, -0.3),
               prop={'size': 8}, fancybox=True, framealpha=0.5, ncol=2)

    plt.ylabel('Episode score')
    plt.xlabel('Episode number')

    if file_to_save_results_graph is not None:
        plt.savefig(file_to_save_results_graph, bbox_inches = "tight")
    plt.show()




