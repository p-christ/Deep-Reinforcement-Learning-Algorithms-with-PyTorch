
import numpy as np
import matplotlib.pyplot as plt

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
    fig, ax = plt.subplots()

    for agent_class in agents:

        agent_name = agent_class.__name__
        rolling_scores = results[agent_name][1]

        episodes_seen = len(rolling_scores)

        if episodes_seen > max_episodes_seen:
            max_episodes_seen = episodes_seen

        ax.plot(range(episodes_seen), rolling_scores)

        legend_values.append(agent_name)    
    
    ax.hlines(y=target_score, xmin=0, xmax=max_episodes_seen, linewidth=2, color='k', linestyles='dotted')    
    
    plt.legend(legend_values, loc='lower right')
    plt.ylabel('Episode score')
    plt.xlabel('Episode number')
    plt.show()          
    