import random

from A3C import A3C
from Agents.DQN_Agents.DQN_HER import DQN_HER
from Agents.DQN_Agents.DDQN import DDQN
from Agents.DQN_Agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from Agents.DQN_Agents.DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets
from Environments.Bit_Flipping_Environment import Bit_Flipping_Environment
from Agents.Policy_Gradient_Agents.PPO import PPO
from Trainer import Trainer
from Utilities.Data_Structures.Config import Config
from Agents.DQN_Agents.DQN import DQN
import numpy as np
import torch

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

config = Config()
config.seed = 1
config.environment = Bit_Flipping_Environment(4)
config.num_episodes_to_run = 2000
config.file_to_save_data_results = None
config.file_to_save_results_graph = None
config.visualise_individual_results = False
config.visualise_overall_agent_results = False
config.randomise_random_seed = False
config.runs_per_agent = 1
config.use_GPU = False
config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.005,
        "batch_size": 64,
        "buffer_size": 40000,
        "epsilon": 0.1,
        "epsilon_decay_rate_denominator": 200,
        "discount_rate": 0.99,
        "tau": 0.1,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.4,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 3,
        "linear_hidden_units": [20, 20, 20],
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "HER_sample_proportion": 0.8
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
        "linear_hidden_units": [20],
        "final_layer_activation": "SOFTMAX",
        "learning_iterations_per_round": 7,
        "discount_rate": 0.99,
        "batch_norm": False,
        "clip_epsilon": 0.1,
        "episodes_per_learning_round": 7,
        "normalise_rewards": False,
        "gradient_clipping_norm": 5,
        "mu": 0.0, #only required for continuous action games
        "theta": 0.0, #only required for continuous action games
        "sigma": 0.0, #only required for continuous action games
        "epsilon_decay_rate_denominator": 1
    },

    "Actor_Critic_Agents": {

        "learning_rate": 0.005,
        "linear_hidden_units": [20, 10],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 50.0,
        "normalise_rewards": True

    }
}


def test_agent_solve_bit_flipping_game():
    AGENTS = [PPO, DDQN, DQN_With_Fixed_Q_Targets, DDQN_With_Prioritised_Experience_Replay, DQN, DQN_HER]
    trainer = Trainer(config, AGENTS)
    results = trainer.run_games_for_agents()
    for agent in AGENTS:
        agent_results = results[agent.agent_name]
        agent_results = np.max(agent_results[0][1][50:])
        assert agent_results >= 0.0, "Failed for {} -- score {}".format(agent.agent_name, agent_results)
