from PPO_Agent import PPO_Agent
from Utilities.Config import Config
from Agents.DQN_Agents.DDQN_Agent import DDQN_Agent
from Agents.DQN_Agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from Agents.DQN_Agents.DQN_Agent import DQN_Agent
from Agents.DQN_Agents.DQN_Agent_With_Fixed_Q_Targets import DQN_Agent_With_Fixed_Q_Targets
from Environments.Open_AI_Gym_Environments.Cart_Pole_Environment import Cart_Pole_Environment
from Agents.Policy_Gradient_Agents.REINFORCE_Agent import REINFORCE_Agent
from Agents.Stochastic_Policy_Search_Agents.Genetic_Agent import Genetic_Agent
from Agents.Stochastic_Policy_Search_Agents.Hill_Climbing_Agent import Hill_Climbing_Agent
from Utilities.Utility_Functions import run_games_for_agents, load_obj, visualise_results_by_agent, save_obj

config = Config()
config.seed = 100
config.environment = Cart_Pole_Environment()
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
        "buffer_size": 20000,
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
        "final_layer_activation": None,
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
        "learning_rate": 0.001,
        "nn_layers": 2,
        "nn_start_units": 20,
        "nn_unit_decay": 1.0,
        "final_layer_activation": "SOFTMAX",
        "learning_iterations_per_round": 10,
        "discount_rate": 0.99,
        "batch_norm": False,
        "clip_epsilon": 0.1,
        "episodes_per_learning_round": 5,
        "normalise_rewards": True
    }
}

AGENTS = [PPO_Agent, DQN_Agent, DQN_Agent_With_Fixed_Q_Targets, DDQN_Agent, DDQN_With_Prioritised_Experience_Replay,
          Genetic_Agent, Hill_Climbing_Agent, REINFORCE_Agent]


run_games_for_agents(config, AGENTS)

