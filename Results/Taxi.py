import gym

from Hierarchical_Agents.SNN_HRL import SNN_HRL
from Agents.Trainer import Trainer
from Utilities.Data_Structures.Config import Config
from Agents.DQN_Agents.DQN import DQN
from Agents.Hierarchical_Agents.h_DQN import h_DQN

config = Config()
config.seed = 1
config.environment = gym.make("Taxi-v2")
config.env_parameters = {}
config.num_episodes_to_run = 10000
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

    "h_DQN": {
        "CONTROLLER": {
            "batch_size": 256,
            "learning_rate": 0.01,
            "buffer_size": 40000,
            "linear_hidden_units": [20, 10],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0, 1],
            "embedding_dimensions": [[config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))],
                                     [config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))]],
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 1500,
            "discount_rate": 0.999,
            "learning_iterations": 1
        },
        "META_CONTROLLER": {
            "batch_size": 256,
            "learning_rate": 0.001,
            "buffer_size": 40000,
            "linear_hidden_units": [20, 10],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": [[config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))]],
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 2500,
            "discount_rate": 0.999,
            "learning_iterations": 1
        }
    },

    "SNN_HRL": {
        "SKILL_AGENT": {
            "num_skills": 2,
            "regularisation_weight": 1.5,
            "visitations_decay": 0.9999,
            "episodes_for_pretraining": 2000,
            "batch_size": 256,
            "learning_rate": 0.01,
            "buffer_size": 40000,
            "linear_hidden_units": [20, 10],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0, 1],
            "embedding_dimensions": [[config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))],
                                     [6, 4]],
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 50,
            "discount_rate": 0.999,
            "learning_iterations": 1
    },

        "MANAGER": {
            "timesteps_before_changing_skill": 4,
            "linear_hidden_units": [10, 5],
            "learning_rate": 0.01,
            "buffer_size": 40000,
            "batch_size": 256,
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": [[config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))]],
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 1000,
            "discount_rate": 0.999,
            "learning_iterations": 1

        }

    }

}

config.hyperparameters["DQN_Agents"] =  config.hyperparameters["h_DQN"]["META_CONTROLLER"]


if __name__ == "__main__":
    AGENTS = [SNN_HRL, DQN, h_DQN]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()



