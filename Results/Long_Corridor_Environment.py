from Trainer import Trainer
from Utilities.Data_Structures.Config import Config
from Agents.DQN_Agents.DQN import DQN
from Agents.Hierarchical_Agents.h_DQN import h_DQN
from Environments.Long_Corridor_Environment import Long_Corridor_Environment

config = Config()
config.seed = 1
config.environment = Long_Corridor_Environment()
config.num_episodes_to_run = 1500
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

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.00025,
        "batch_size": 128,
        "buffer_size": 40000,
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 1000,
        "discount_rate": 0.99,
        "tau": 0.1,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 3,
        "linear_hidden_units": [10, 10],
        "final_layer_activation": "None",
        "columns_of_data_to_be_embedded": [0],
        "embedding_dimensions": [[config.environment.get_num_possible_states(), max(4, int(config.environment.get_num_possible_states() / 10.0))]],
        "batch_norm": False,
        "gradient_clipping_norm": 5
    },


    "h_DQN": {

        "CONTROLLER": {
            "batch_size": 128,
            "learning_rate": 0.00025,
            "buffer_size": 30000,
            "linear_hidden_units": [10, 10],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0, 1],
            "embedding_dimensions": [[config.environment.get_num_possible_states(),
                                      max(4, int(config.environment.get_num_possible_states() / 10.0))],
                                     [config.environment.get_num_possible_states(),
                                      max(4, int(config.environment.get_num_possible_states() / 10.0))]
                                     ],
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_denominator": 1


        },

        "META_CONTROLLER": {

            "batch_size": 128,
            "learning_rate": 0.00025,
            "buffer_size": 30000,
            "linear_hidden_units": [10, 10],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": [[config.environment.get_num_possible_states(),
                                      max(4, int(config.environment.get_num_possible_states() / 10.0))]],
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_denominator": 1

        }

    }

}

if __name__ == "__main__":
    AGENTS = [h_DQN]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()






