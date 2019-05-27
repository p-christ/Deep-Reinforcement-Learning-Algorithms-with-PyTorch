

# Key questions --> a lot of pre training or a little

# add checks everywhere first that all the levers work!

learning_rate = 0.001
buffer_size = 100000
batch_size = 256
batch_norm = False
embedding_dimensionality = 10
update_every_n_steps = 1
learning_iterations = 1
discount_rate = 0.99
tau = 0.01
sequitur_k = 2

pre_training_learning_iterations_multipliers = [100, 500, 1000, 2000]
episodes_to_run_with_no_explorations = [0, 5, 10, 20]
epsilon_decay_rate_denominators = [30, 50, 100, 150, 200] #150
min_num_episodes_to_plays = [5, 20, 50, 100] #80
gradient_clipping_norms = [0.2, 0.5, 1.0]  #needs to be optimised
linear_hidden_unitss = [[32, 32], [21, 21, 21], [16, 16, 16, 16]]

num_top_results_to_uses = [2, 5, 10, 20]
action_frequency_required_in_top_resultss = [0.2, 0.5, 0.7, 0.9]

random_episodes_to_run = 0

action_length_reward_bonuss = [0.05, 0.1, 0.2, 0.4, 0.8]
use_global_list_of_best_performing_actionss = [True, False]

keep_previous_output_layers = [True, False]
only_train_new_actionss = [True, False]
only_train_final_layers = [True, False]
reduce_macro_action_appearance_cutoff_throughout_trainings = [True, False]

add_1_macro_action_at_a_times = [True, False]
calculate_q_values_as_incrementss = [True, False]
action_balanced_replay_buffers = [True, False]
copy_over_hidden_layerss = [True, False]
increase_batch_size_with_actionss = [True, False]


for pre_training_learning_iterations_multiplier in pre_training_learning_iterations_multipliers:
    for episodes_to_run_with_no_exploration in episodes_to_run_with_no_explorations:
        for epsilon_decay_rate_denominator in epsilon_decay_rate_denominators:
            for min_num_episodes_to_play in min_num_episodes_to_plays:
                for gradient_clipping_norm in gradient_clipping_norms:
                    for linear_hidden_units in linear_hidden_unitss:
                        for num_top_results_to_use in num_top_results_to_uses:
                            for action_frequency_required_in_top_results in action_frequency_required_in_top_resultss:
                                for action_length_reward_bonus in action_length_reward_bonuss:
                                    for use_global_list_of_best_performing_actions in use_global_list_of_best_performing_actionss:
                                        for keep_previous_output_layer in keep_previous_output_layers:
                                            for only_train_new_actions in only_train_new_actionss:
                                                for only_train_final_layer in only_train_final_layers:
                                                    for reduce_macro_action_appearance_cutoff_throughout_training in reduce_macro_action_appearance_cutoff_throughout_trainings:
                                                        for add_1_macro_action_at_a_time in add_1_macro_action_at_a_times:
                                                            for calculate_q_values_as_increments in calculate_q_values_as_incrementss:
                                                                for action_balanced_replay_buffer in action_balanced_replay_buffers:
                                                                    for copy_over_hidden_layers in copy_over_hidden_layerss:
                                                                        for increase_batch_size_with_actions in increase_batch_size_with_actionss:
                                                                            config.hyperparameters = {

                                                                                "HRL": {
                                                                                    "linear_hidden_units": linear_hidden_units,
                                                                                    "learning_rate": learning_rate,
                                                                                    "buffer_size": buffer_size,
                                                                                    "batch_size": batch_size,
                                                                                    "final_layer_activation": "None",
                                                                                    "columns_of_data_to_be_embedded": [0],
                                                                                    "embedding_dimensions": [[
                                                                                                                 config.environment.observation_space.n,
                                                                                                                 embedding_dimensionality]],
                                                                                    "batch_norm": batch_norm,
                                                                                    "gradient_clipping_norm": gradient_clipping_norm,
                                                                                    "update_every_n_steps": update_every_n_steps,
                                                                                    "epsilon_decay_rate_denominator": epsilon_decay_rate_denominator,
                                                                                    "discount_rate": discount_rate,
                                                                                    "learning_iterations": learning_iterations,
                                                                                    "tau": tau,
                                                                                    "sequitur_k": sequitur_k,
                                                                                    "action_length_reward_bonus": action_length_reward_bonus,
                                                                                    "pre_training_learning_iterations_multiplier": pre_training_learning_iterations_multiplier,
                                                                                    "episodes_to_run_with_no_exploration": episodes_to_run_with_no_exploration,
                                                                                    "action_balanced_replay_buffer": action_balanced_replay_buffer,
                                                                                    "copy_over_hidden_layers": copy_over_hidden_layers,
                                                                                    "use_global_list_of_best_performing_actions": use_global_list_of_best_performing_actions,
                                                                                    "keep_previous_output_layer": keep_previous_output_layer,
                                                                                    "random_episodes_to_run": random_episodes_to_run,
                                                                                    "only_train_new_actions": only_train_new_actions,
                                                                                    "only_train_final_layer": only_train_final_layer,
                                                                                    "num_top_results_to_use": num_top_results_to_use,
                                                                                    "action_frequency_required_in_top_results": action_frequency_required_in_top_results,
                                                                                    "reduce_macro_action_appearance_cutoff_throughout_training": reduce_macro_action_appearance_cutoff_throughout_training,
                                                                                    "add_1_macro_action_at_a_time": add_1_macro_action_at_a_time,
                                                                                    "calculate_q_values_as_increments": calculate_q_values_as_increments,
                                                                                    "min_num_episodes_to_play": min_num_episodes_to_play,
                                                                                    "increase_batch_size_with_actions": increase_batch_size_with_actions
                                                                                }}





config.hyperparameters = {

    "HRL": {
        "linear_hidden_units": linear_hidden_units,
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "final_layer_activation": "None",
        "columns_of_data_to_be_embedded": [0],
        "embedding_dimensions": [[config.environment.observation_space.n, embedding_dimensionality]],
        "batch_norm": batch_norm,
        "gradient_clipping_norm": gradient_clipping_norm,
        "update_every_n_steps": update_every_n_steps,
        "epsilon_decay_rate_denominator": epsilon_decay_rate_denominator,
        "discount_rate": discount_rate,
        "learning_iterations": learning_iterations,
        "tau": tau,
        "sequitur_k": sequitur_k,
        "action_length_reward_bonus": action_length_reward_bonus,
        "pre_training_learning_iterations_multiplier": pre_training_learning_iterations_multiplier,
        "episodes_to_run_with_no_exploration": episodes_to_run_with_no_exploration,
        "action_balanced_replay_buffer": action_balanced_replay_buffer,
        "copy_over_hidden_layers": copy_over_hidden_layers,
        "use_global_list_of_best_performing_actions": use_global_list_of_best_performing_actions,
        "keep_previous_output_layer": keep_previous_output_layer,
        "random_episodes_to_run": random_episodes_to_run,
        "only_train_new_actions": only_train_new_actions,
        "only_train_final_layer": only_train_final_layer,
        "num_top_results_to_use": num_top_results_to_use,
        "action_frequency_required_in_top_results": action_frequency_required_in_top_results,
        "reduce_macro_action_appearance_cutoff_throughout_training": reduce_macro_action_appearance_cutoff_throughout_training,
        "add_1_macro_action_at_a_time": add_1_macro_action_at_a_time,
        "calculate_q_values_as_increments": calculate_q_values_as_increments,
        "min_num_episodes_to_play": min_num_episodes_to_play,
        "increase_batch_size_with_actions": increase_batch_size_with_actions
    },