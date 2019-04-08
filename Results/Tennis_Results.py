from Actor_Critic_Agents.DDPG_Agent import DDPG_Agent
from Data_Structures.Config import Config
from Unity_Environments.Tennis_Environment import Tennis_Environment
from Utility_Functions import run_games_for_agents

config = Config()
config.seed = 100
config.environment = Tennis_Environment()
config.max_episodes_to_run = 10000
config.file_to_save_data_results = "Results_Data.pkl"
config.file_to_save_data_results_graph = "Results_Graph.png"
config.visualise_individual_results = True
config.visualise_overall_results = True
config.runs_per_agent = 1
config.use_GPU = False

config.hyperparameters = {

    "Actor_Critic_Agents": {
        "Actor": {
            "learning_rate": 0.0001,
            "nn_layers": 3,
            "nn_start_units": 128,
            "nn_unit_decay": 0.5,
            "final_layer_activation": "TANH",
            "batch_norm": False,
            "tau": 0.05,
            "update_every_n_steps": 10,
            "gradient_clipping_norm": 5
        },

        "Critic": {
            "learning_rate": 0.0003,
            "nn_layers": 3,
            "nn_start_units": 128,
            "nn_unit_decay": 0.5,
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 100000,
            "tau": 0.05,
            "update_every_n_steps": 10,
            "gradient_clipping_norm": 5
        },

        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0,
        "theta": 0.15,
        "sigma": 0.2,
        "learning_updates_per_learning_session": 10
    }
}

if __name__== '__main__':
    AGENTS = [DDPG_Agent]
    run_games_for_agents(config, AGENTS)