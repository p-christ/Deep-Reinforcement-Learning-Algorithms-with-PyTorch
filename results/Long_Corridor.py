from agents.hierarchical_agents.SNN_HRL import SNN_HRL
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from agents.DQN_agents.DQN import DQN
from agents.hierarchical_agents.h_DQN import h_DQN
from environments.Long_Corridor_Environment import Long_Corridor_Environment

config = Config()
config.seed = 1
config.env_parameters = {"stochasticity_of_action_right": 0.5}
config.environment = Long_Corridor_Environment(stochasticity_of_action_right=config.env_parameters["stochasticity_of_action_right"])
config.num_episodes_to_run = 10000
config.file_to_save_data_results = "Data_and_Graphs/Long_Corridor_Results_Data.pkl"
config.file_to_save_results_graph = "Data_and_Graphs/Long_Corridor_Results_Graph.png"
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
            "visitations_decay": 0.99,
            "episodes_for_pretraining": 2000,
            # "batch_size": 256,
            # "learning_rate": 0.01,
            # "buffer_size": 40000,
            # "linear_hidden_units": [20, 10],
            # "final_layer_activation": "None",
            # "columns_of_data_to_be_embedded": [0, 1],
            # "embedding_dimensions": [[config.environment.observation_space.n,
            #                           max(4, int(config.environment.observation_space.n / 10.0))],
            #                          [6, 4]],
            # "batch_norm": False,
            # "gradient_clipping_norm": 5,
            # "update_every_n_steps": 1,
            # "epsilon_decay_rate_denominator": 50,
            # "discount_rate": 0.999,
            # "learning_iterations": 1


            "learning_rate": 0.05,
            "linear_hidden_units": [20, 20],
            "final_layer_activation": "SOFTMAX",
            "learning_iterations_per_round": 5,
            "discount_rate": 0.99,
            "batch_norm": False,
            "clip_epsilon": 0.1,
            "episodes_per_learning_round": 4,
            "normalise_rewards": True,
            "gradient_clipping_norm": 7.0,
            "mu": 0.0,  # only required for continuous action games
            "theta": 0.0,  # only required for continuous action games
            "sigma": 0.0,  # only required for continuous action games
            "epsilon_decay_rate_denominator": 1.0



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



