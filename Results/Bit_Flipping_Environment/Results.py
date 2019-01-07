from DQN_HER_Agent import DQN_HER_Agent
from Other_Enrivonments.Bit_Flipping_Environment import Bit_Flipping_Environment
from PPO_Agent import PPO_Agent
from Data_Structures.Config import Config
from Agents.DQN_Agents.DDQN_Agent import DDQN_Agent
from Agents.DQN_Agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from Agents.DQN_Agents.DQN_Agent import DQN_Agent
from Agents.DQN_Agents.DQN_Agent_With_Fixed_Q_Targets import DQN_Agent_With_Fixed_Q_Targets
from Agents.Policy_Gradient_Agents.REINFORCE_Agent import REINFORCE_Agent
from Agents.Stochastic_Policy_Search_Agents.Genetic_Agent import Genetic_Agent
from Agents.Stochastic_Policy_Search_Agents.Hill_Climbing_Agent import Hill_Climbing_Agent
from Utilities.Utility_Functions import run_games_for_agents

config = Config()
config.seed = 100
config.environment = Bit_Flipping_Environment()
config.max_episodes_to_run = 500
config.file_to_save_data_results = "Results_Data.pkl"
config.file_to_save_data_results_graph = "Results_Graph.png"
config.visualise_individual_results = True
config.visualise_overall_results = True
config.runs_per_agent = 10

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.005,
        "batch_size": 256,
        "buffer_size": 40000,
        "epsilon": 0.1,
        "epsilon_decay_rate_denominator": 200,
        "discount_rate": 0.99,
        "tau": 0.1,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.4,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "nn_layers": 3,
        "nn_start_units": 20,
        "nn_unit_decay": 1.0,
        "final_layer_activation": None,
        "batch_norm": False
    }
}

AGENTS = [DQN_HER_Agent, DQN_Agent]


run_games_for_agents(config, AGENTS)

