import gym

from A2C import A2C
from Agents.Actor_Critic_Agents.A3C import A3C
from Agents.Policy_Gradient_Agents.PPO import PPO
from Agents.Trainer import Trainer
from Hierarchical_Agents.HIRO import HIRO
from Utilities.Data_Structures.Config import Config
from Agents.DQN_Agents.DDQN import DDQN
from Agents.DQN_Agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from Agents.DQN_Agents.DQN import DQN
from Agents.DQN_Agents.DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets

config = Config()
config.seed = 1
config.environment = gym.make("Pendulum-v0")
config.num_episodes_to_run = 1500
config.file_to_save_data_results = None
config.file_to_save_results_graph = None
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False




config.hyperparameters = {
    "HIRO": {

        "LOWER_LEVEL": {
            "max_lower_level_timesteps": 3,

            "Actor": {
                "learning_rate": 0.001,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": "TANH",
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5
            },

            "Critic": {
                "learning_rate": 0.01,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": "None",
                "batch_norm": False,
                "buffer_size": 100000,
                "tau": 0.005,
                "gradient_clipping_norm": 5
            },

            "batch_size": 256,
            "discount_rate": 0.9,
            "mu": 0.0,  # for O-H noise
            "theta": 0.15,  # for O-H noise
            "sigma": 0.25,  # for O-H noise
            "action_noise_std": 0.2,  # for TD3
            "action_noise_clipping_range": 0.5,  # for TD3
            "update_every_n_steps": 20,
            "learning_updates_per_learning_session": 10,

            } ,



        "HIGHER_LEVEL": {

                        "Actor": {
                "learning_rate": 0.001,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": "TANH",
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5
            },

            "Critic": {
                "learning_rate": 0.01,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": "None",
                "batch_norm": False,
                "buffer_size": 100000,
                "tau": 0.005,
                "gradient_clipping_norm": 5
            },

            "batch_size": 256,
            "discount_rate": 0.9,
            "mu": 0.0,  # for O-H noise
            "theta": 0.15,  # for O-H noise
            "sigma": 0.25,  # for O-H noise
            "action_noise_std": 0.2,  # for TD3
            "action_noise_clipping_range": 0.5,  # for TD3
            "update_every_n_steps": 20,
            "learning_updates_per_learning_session": 10,

            } ,


        }


    }


if __name__ == "__main__":
    AGENTS = [HIRO]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()






