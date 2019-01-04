from Actor_Critic_Agents.DDPG_Agent import DDPG_Agent
from Config import Config
from Fetch_Reach_Environment import Fetch_Reach_Environment
from Open_AI_Gym_Environments.Mountain_Car_Continuous_Environment import Mountain_Car_Continuous_Environment
from Utility_Functions import run_games_for_agents

config = Config()
config.seed = 100
config.environment = Fetch_Reach_Environment()
config.max_episodes_to_run = 3000
config.file_to_save_data_results = "Results_Data.pkl"
config.file_to_save_data_results_graph = "Results_Graph.png"
config.visualise_individual_results = False
config.visualise_overall_results = True
config.runs_per_agent = 3

config.hyperparameters = {

    "Actor_Critic_Agents": {
        "Actor": {
            "learning_rate": 0.001,
            "nn_layers": 3,
            "nn_start_units": 256,
            "nn_unit_decay": 1.0,
            "final_layer_activation": "TANH",
            "batch_norm": False,
            "tau": 0.1,
            "update_every_n_steps": 800
        },

        "Critic": {
            "learning_rate": 0.001,
            "nn_layers": 3,
            "nn_start_units": 256,
            "nn_unit_decay": 1.0,
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 10000000,
            "tau": 0.1,
            "update_every_n_steps": 800
        },

        "batch_size": 128,
        "discount_rate": 0.98,
        "mu": 0.0,
        "theta": 0.15,
        "sigma": 0.20,
        "learning_updates_per_learning_session": 40
    }
}

AGENTS = [DDPG_Agent]

run_games_for_agents(config, AGENTS)
