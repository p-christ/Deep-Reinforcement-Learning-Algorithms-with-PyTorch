from Agents.Trainer import Trainer
from DDPG import DDPG
from Hierarchical_Agents.HIRO import HIRO
from Utilities.Data_Structures.Config import Config
import gym

config = Config()
config.seed = 1
config.environment = gym.make("Ant-v2")
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

DDPG_hyperparameters =  {  # hyperparameters taken from https://arxiv.org/pdf/1802.09477.pdf
        "Actor": {
            "learning_rate": 0.001,
            "linear_hidden_units": [400, 300],
            "final_layer_activation": "TANH",
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "Critic": {
            "learning_rate": 0.01,
            "linear_hidden_units": [400, 300],
            "final_layer_activation": "None",
            "batch_norm": False,
            "buffer_size": 100000,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "batch_size": 64,
        "discount_rate": 0.99,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.2,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,

    }

config.hyperparameters = {
    "HIRO": {

        "LOWER_LEVEL": DDPG_hyperparameters ,  # "max_lower_level_timesteps": 5,
        "HIGHER_LEVEL": DDPG_hyperparameters },



    "Actor_Critic_Agents": DDPG_hyperparameters

        }


# ADD Y_RANGE  - 30 to 30...


if __name__ == "__main__":

    #
    AGENTS = [DDPG, HIRO]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()






