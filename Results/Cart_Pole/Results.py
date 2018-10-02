from Utilities.Config import Config
from Agents.DQN_Agents.DDQN_Agent import DDQN_Agent
from Agents.DQN_Agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from Agents.DQN_Agents.DQN_Agent import DQN_Agent
from Agents.DQN_Agents.DQN_Agent_With_Fixed_Q_Targets import DQN_Agent_With_Fixed_Q_Targets
from Environments.Open_AI_Gym_Environments.Cart_Pole_Environment import Cart_Pole_Environment
from Agents.Policy_Gradient_Agents.PPO_Agent import PPO_Agent
from Agents.Policy_Gradient_Agents.REINFORCE_Agent import REINFORCE_Agent
from Agents.Stochastic_Policy_Search_Agents.Genetic_Agent import Genetic_Agent
from Agents.Stochastic_Policy_Search_Agents.Hill_Climbing_Agent import Hill_Climbing_Agent
from Utilities.Utility_Functions import run_games_for_agents

config = Config()
config.seed = 100
config.environment = Cart_Pole_Environment()
config.requirements_to_solve_game = {"average_score_required": 195, "rolling_score_window": 100}
config.max_episodes_to_run = 20
config.file_to_save_data_results = "Results_Data51.pkl"
config.file_to_save_data_results_graph = "Results_Graph.png"
config.visualise_individual_results = False
config.visualise_overall_results = True
config.runs_per_agent = 1

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.005,
        "batch_size": 64,
        "buffer_size": 10000,
        "epsilon": 0.1,
        "discount_rate": 0.99,
        "tau": 0.1,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.4,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "nn_layers": 3,
        "nn_start_units": 20,
        "nn_unit_decay": 1.0,
        "softmax_final_layer": False,
        "batch_norm": False
    },
    "Stochastic_Policy_Search_Agents": {
        "policy_network_type": "Linear",
        "noise_scale_start": 1e-2,
        "noise_scale_min": 1e-3,
        "noise_scale_max": 2.0,
        "noise_scale_growth_factor": 2.0,
        "stochastic_action_decision": False,
        "num_policies": 10,
        "episodes_per_policy": 1,
        "num_policies_to_keep": 5
    },
    "Policy_Gradient_Agents": {
        "learning_rate": 0.01,
        "nn_layers": 1,
        "nn_start_units": 20,
        "nn_unit_decay": 1.0,
        "softmax_final_layer": True,
        "discount_rate": 0.99,
        "batch_norm": False
    }
}

AGENTS = [Genetic_Agent, Hill_Climbing_Agent, REINFORCE_Agent,
          DQN_Agent, DDQN_With_Prioritised_Experience_Replay, DQN_Agent_With_Fixed_Q_Targets, DDQN_Agent,
          REINFORCE_Agent, PPO_Agent]

AGENTS = [DQN_Agent]


run_games_for_agents(config, AGENTS)

