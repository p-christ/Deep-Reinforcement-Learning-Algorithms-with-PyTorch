import matplotlib.pyplot as plt

from Utility_Functions import load_obj, draw_horizontal_line_with_label, hide_spines, set_graph_axis_limits, \
    set_graph_labels

file_to_save_results_graph = "My_diagram"
target_score = 0


results = load_obj("Results_Data.pkl")

agents = results.keys()

fig, axes = plt.subplots(1, 1, sharex=False, figsize = (5, 4.5)) # plt.subplots()

lines = []

for agent_name in agents:

    rolling_scores = results[agent_name][1]
    rolling_scores = rolling_scores[:4502]

    lines.append(rolling_scores)

    episodes_seen = len(rolling_scores)

    time_taken = results[agent_name][2]
    starting_point = time_taken / len(rolling_scores)
    time_axes = [starting_point * (t + 1.0) for t in range(len(rolling_scores))]

    axes.plot(range(episodes_seen), rolling_scores)



max_episodes_seen_by_any_agent = 4502
max_time_taken_by_any_agent =  max([results[agent_name][2] for agent_name in agents])

min_score_achieved_by_any_agent = min([min(rolling_scores) for rolling_scores in [results[agent_name][1] for agent_name in agents]]) - 1.0

draw_horizontal_line_with_label(axes, y_value=target_score, x_min=0, x_max=max_episodes_seen_by_any_agent*1.001, label="Target \n score")

hide_spines(axes, ['right', 'top'])

set_graph_axis_limits(axes, 0, 4502, min_score_achieved_by_any_agent, target_score)


plt.figlegend(lines, labels=agents, loc='lower center', ncol=3, labelspacing=0.)

axes.set_title("Score vs. Episodes Played", y=1.03, fontweight='bold')

set_graph_labels(axes, xlabel='Rolling Episode Scores', ylabel='Episode number')

if file_to_save_results_graph is not None:
    plt.savefig(file_to_save_results_graph)
plt.show()
