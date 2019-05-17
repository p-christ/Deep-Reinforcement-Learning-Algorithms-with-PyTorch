import gym

from deep_reinforcement_learning_algorithms_with_pytorch.agents.DQN_agents.DDQN import DDQN
from deep_reinforcement_learning_algorithms_with_pytorch.agents.hierarchical_agents.HRL import HRL
from deep_reinforcement_learning_algorithms_with_pytorch.agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from deep_reinforcement_learning_algorithms_with_pytorch.agents.hierarchical_agents.SNN_HRL import SNN_HRL
from deep_reinforcement_learning_algorithms_with_pytorch.agents.Trainer import Trainer
from deep_reinforcement_learning_algorithms_with_pytorch.utilities.data_structures.Config import Config
from deep_reinforcement_learning_algorithms_with_pytorch.agents.DQN_agents.DQN import DQN
from deep_reinforcement_learning_algorithms_with_pytorch.agents.hierarchical_agents.h_DQN import h_DQN

config = Config()
config.seed = 1
config.environment = gym.make("Taxi-v2")
config.env_parameters = {}
config.num_episodes_to_run = 1000
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


linear_hidden_units = [64, 32]
learning_rate = 0.01
buffer_size = 100000
batch_size = 256
batch_norm = False
embedding_dimensionality = 15
gradient_clipping_norm = 5
update_every_n_steps = 1
learning_iterations = 1
epsilon_decay_rate_denominator = 50
discount_rate = 0.99
tau = 0.01
sequitur_k = 10

config.hyperparameters = {

    "HRL": {
        "linear_hidden_units": linear_hidden_units,
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "final_layer_activation": "None",
        "columns_of_data_to_be_embedded": [0],
        "embedding_dimensions": [[config.environment.observation_space.n, embedding_dimensionality]],
        "batch_norm": batch_norm,
        "gradient_clipping_norm": gradient_clipping_norm,
        "update_every_n_steps": update_every_n_steps,
        "epsilon_decay_rate_denominator": epsilon_decay_rate_denominator,
        "discount_rate": discount_rate,
        "learning_iterations": learning_iterations,
        "tau": tau,
        "sequitur_k": sequitur_k,
        "action_length_reward_bonus": 0.1
    },

    "DQN_Agents": {
        "linear_hidden_units": linear_hidden_units,
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "final_layer_activation": "None",
        "columns_of_data_to_be_embedded": [0],
        "embedding_dimensions": [[config.environment.observation_space.n, embedding_dimensionality]],
        "batch_norm": batch_norm,
        "gradient_clipping_norm": gradient_clipping_norm,
        "update_every_n_steps": update_every_n_steps,
        "epsilon_decay_rate_denominator": epsilon_decay_rate_denominator,
        "discount_rate": discount_rate,
        "learning_iterations": learning_iterations,
        "tau": tau,
    },

    "Actor_Critic_Agents": {
        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "Softmax",
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": [[config.environment.observation_space.n, embedding_dimensionality]],
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": [[config.environment.observation_space.n, embedding_dimensionality]],
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 400,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    }
}


if __name__ == "__main__":
    AGENTS = [HRL] #SAC_Discrete, DDQN] #HRL] #, SNN_HRL, DQN, h_DQN]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()



