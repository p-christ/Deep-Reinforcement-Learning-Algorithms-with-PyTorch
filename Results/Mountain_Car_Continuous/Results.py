from Actor_Critic_Agents.DDPG_Agent import DDPG_Agent
from Data_Structures.Config import Config
from Open_AI_Gym_Environments.Mountain_Car_Continuous_Environment import Mountain_Car_Continuous_Environment
from PPO_Agent import PPO_Agent
from Utility_Functions import run_games_for_agents

config = Config()
config.seed = 100
config.environment = Mountain_Car_Continuous_Environment()
config.max_episodes_to_run = 3000
config.file_to_save_data_results = "Results_Data3.pkl"
config.file_to_save_data_results_graph = "Results_Graph2.png"
config.visualise_individual_results = False
config.visualise_overall_results = True
config.runs_per_agent = 2
config.use_GPU = False

config.hyperparameters = {
    "Policy_Gradient_Agents": {
            "learning_rate": 0.02,
            "nn_layers": 2,
            "nn_start_units": 20,
            "nn_unit_decay": 1.0,
            "final_layer_activation": None,
            "learning_iterations_per_round": 10,
            "discount_rate": 0.99,
            "batch_norm": False,
            "clip_epsilon": 0.2,
            "episodes_per_learning_round": 7,
            "normalise_rewards": True,
            "gradient_clipping_norm": 5,
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.25
        },

    "Actor_Critic_Agents": {
        "Actor": {
            "learning_rate": 0.001,
            "nn_layers": 5,
            "nn_start_units": 50,
            "nn_unit_decay": 1.0,
            "final_layer_activation": "TANH",
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "Critic": {
            "learning_rate": 0.01,
            "nn_layers": 6,
            "nn_start_units": 50,
            "nn_unit_decay": 1.0,
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 30000,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "batch_size": 256,
        "discount_rate": 0.9,
        "mu": 0.0,
        "theta": 0.15,
        "sigma": 0.25, #0.22 did well before
        "update_every_n_steps": 10,
        "learning_updates_per_learning_session": 10
    }
}

if __name__== '__main__':
    AGENTS = [PPO_Agent, DDPG_Agent]
    run_games_for_agents(config, AGENTS)
