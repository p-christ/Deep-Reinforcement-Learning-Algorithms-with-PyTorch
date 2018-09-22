from DQN_Agents.DDQN_Agent import DDQN_Agent
from DQN_Agents.DQN_Agent import DQN_Agent
from DQN_Agents.DQN_Agent_With_Fixed_Q_Targets import DQN_Agent_With_Fixed_Q_Targets
from Open_AI_Gym_Environments.Cart_Pole_Environment import Cart_Pole_Environment
from Policy_Gradient_Agents.REINFORCE_Agent import REINFORCE_Agent
from Stochastic_Policy_Search_Agents.Genetic_Agent import Genetic_Agent
from Stochastic_Policy_Search_Agents.Hill_Climbing_Agent import Hill_Climbing_Agent
from Utilities.Utility_Functions import run_games_for_agents

ENVIRONMENT = Cart_Pole_Environment()
REQUIREMENTS_TO_SOLVE_GAME = {"average_score_required": 195, "rolling_score_window": 100}
MAX_EPISODES_TO_RUN = 5000
FILE_TO_SAVE_DATA_RESULTS = "Results_Data.pkl"
FILE_TO_SAVE_RESULTS_GRAPH = "Results_Graph.png"
RUNS_PER_AGENT = 10
SEED = 100

AGENTS = [Genetic_Agent, Hill_Climbing_Agent, REINFORCE_Agent,
          DQN_Agent, DQN_Agent_With_Fixed_Q_Targets, DDQN_Agent]

AGENTS = [DQN_Agent, DQN_Agent_With_Fixed_Q_Targets, DDQN_Agent]

hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.0006,
        "batch_size": 64,
        "buffer_size": 1000000,
        "epsilon": 0.1,
        "discount_rate": 0.99,
        "tau": 1e-3,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.4,
        "update_every_n_steps": 1,
        "nn_layers": 3,
        "nn_start_units": 20,
        "nn_unit_decay": 1.0,
        "softmax_final_layer": False
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
        "discount_rate": 0.99
    }
}

run_games_for_agents(ENVIRONMENT, AGENTS, RUNS_PER_AGENT, hyperparameters, REQUIREMENTS_TO_SOLVE_GAME,
                     MAX_EPISODES_TO_RUN, visualise_results=True, save_data_filename=FILE_TO_SAVE_DATA_RESULTS,
                     file_to_save_results_graph=FILE_TO_SAVE_RESULTS_GRAPH, seed=SEED)

