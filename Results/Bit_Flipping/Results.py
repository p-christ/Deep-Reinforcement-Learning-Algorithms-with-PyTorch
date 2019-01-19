from Agents.DQN_Agents.DQN_HER_Agent import DQN_HER_Agent
from Environments.Other_Enrivonments.Bit_Flipping_Environment import Bit_Flipping_Environment
from Utilities.Data_Structures.Config import Config
from Agents.DQN_Agents.DQN_Agent import DQN_Agent
from Utilities.Utility_Functions import run_games_for_agents

config = Config()
config.seed = 100
config.environment = Bit_Flipping_Environment(14)
config.max_episodes_to_run = 6000
config.file_to_save_data_results = "Results_Data.pkl"
config.file_to_save_data_results_graph = "Results_Graph.png"
config.visualise_individual_results = True
config.visualise_overall_results = True
config.runs_per_agent = 3
config.use_GPU = False

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.001,
        "batch_size": 128,
        "buffer_size": 100000,
        "epsilon": 0.1,
        "epsilon_decay_rate_denominator": 500,
        "discount_rate": 0.98,
        "tau": 0.1,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.4,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "nn_layers": 2,
        "nn_start_units": 256,
        "nn_unit_decay": 1.0,
        "final_layer_activation": None,
        "batch_norm": False,
        "gradient_clipping_norm": 5
    }
}

if __name__== '__main__':
    AGENTS = [DQN_HER_Agent, DQN_Agent]
    run_games_for_agents(config, AGENTS)

