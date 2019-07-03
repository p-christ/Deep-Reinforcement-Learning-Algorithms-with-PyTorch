import gym
import pytest


from utilities.Utility_Functions import flatten_action_id_to_actions
from utilities.data_structures.Config import Config

config = Config()
config.seed = 1
config.environment = gym.make("Taxi-v2")
config.env_parameters = {}
config.num_episodes_to_run = 1000
config.file_to_save_data_results = None
config.file_to_save_results_graph = None
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 3
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False

linear_hidden_units = [10, 5]
learning_rate = 0.01
buffer_size = 40000
batch_size = 256
batch_norm = False
embedding_dimensionality = 15
gradient_clipping_norm = 5
update_every_n_steps = 1
learning_iterations = 1
epsilon_decay_rate_denominator = 400
discount_rate = 0.99
tau = 0.01
sequitur_k = 10

config.hyperparameters = {


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
        "action_length_reward_bonus": 0.1,
        "episodes_to_run_with_no_exploration": 10,
        "pre_training_learning_iterations_multiplier": 0.1,
        "copy_over_hidden_layers": True,
        "use_global_list_of_best_performing_actions": True
}


# hrl = HRL(config)

# def test_flatten_action_id_to_actions():
#     """Tests flatten_action_id_to_actions"""
#     action_id_to_actions = {0: (0,), 1:(1,), 2:(0, 1), 3: (2, 1), 4:(2, 3)}
#     original_number_of_primitive_actions = 2
#
#
#
#     flattened_action_id_to_actions = flatten_action_id_to_actions(action_id_to_actions, original_number_of_primitive_actions)
#     assert flattened_action_id_to_actions == {0: (0,), 1:(1,), 2:(0, 1), 3: (0, 1, 1), 4:(0, 1, 0, 1, 1)}, flattened_action_id_to_actions
#
#     action_id_to_actions = {0: (0,), 1:(1,), 2:(2,)}
#     original_number_of_primitive_actions = 3
#     flattened_action_id_to_actions = flatten_action_id_to_actions(action_id_to_actions, original_number_of_primitive_actions)
#     assert flattened_action_id_to_actions == action_id_to_actions
#
#     with pytest.raises(AssertionError):
#         action_id_to_actions = {0: (0,), 1: (1,), 2: (2,)}
#         original_number_of_primitive_actions = 4
#         flattened_action_id_to_actions = flatten_action_id_to_actions(action_id_to_actions,
#                                                                           original_number_of_primitive_actions)
#     with pytest.raises(AssertionError):
#         action_id_to_actions = {0: (0,), 1: (1,), 2: (2, 2)}
#         original_number_of_primitive_actions = 3
#         flattened_action_id_to_actions = flatten_action_id_to_actions(action_id_to_actions,
#                                                                           original_number_of_primitive_actions)

