
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


def override(parent_class):
    def overrider(method):
        assert method.__name__ in dir(parent_class), "You aren't overriding anything"
        return method
    return overrider

def print_two_lines():
    print("-----------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------") 
    print(" ")
    
    
def save_score_results(file_path, results):
    np.save(file_path, results) 
    
    
def visualise_results_by_agent(agents, results, target_score):


    legend_values = []
    max_episodes_seen = 0
    min_score_seen = float("inf")
    fig, ax = plt.subplots()

    for agent_class in agents:

        agent_name = agent_class.__name__
        rolling_scores = results[agent_name][1]

        episodes_seen = len(rolling_scores)

        if episodes_seen > max_episodes_seen:
            max_episodes_seen = episodes_seen

        min_score = min(rolling_scores)
        if min_score < min_score_seen:
            min_score_seen = min_score

        ax.plot(range(episodes_seen), rolling_scores)

        legend_values.append(agent_name)    
    
    ax.hlines(y=target_score, xmin=0, xmax=max_episodes_seen,
              linewidth=2, color='k', linestyles='dotted')

    ax.text(-55, target_score * 0.95, "Target \n score")

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.xlim(xmin=0)
    plt.ylim(ymin=min_score_seen)
    plt.ylim(ymax=target_score)

    plt.legend(legend_values, loc='center', bbox_to_anchor=(0.5, -0.2),
               prop={'size': 8}, fancybox=True, framealpha=0.5, ncol=2)

    plt.ylabel('Episode score')
    plt.xlabel('Episode number')
    plt.show()          


