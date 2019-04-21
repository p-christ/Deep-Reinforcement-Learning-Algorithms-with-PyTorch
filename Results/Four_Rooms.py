from Agents.DQN_Agents.DQN_HER import DQN_HER
from DDQN import DDQN
from Environments.Four_Rooms_Environment import Four_Rooms_Environment
from Hierarchical_Agents.SNN_HRL import SNN_HRL
from Trainer import Trainer
from Utilities.Data_Structures.Config import Config
from Agents.DQN_Agents.DQN import DQN

config = Config()
config.seed = 1
config.environment = Four_Rooms_Environment(15, 15, stochastic_actions_probability=0.25, random_start_user_place=True, random_goal_place=False)

config.num_episodes_to_run = 1000
config.file_to_save_data_results = "Data_and_Graphs/Four_Rooms.pkl"
config.file_to_save_results_graph = "Data_and_Graphs/Four_Rooms.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 3
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False


config.hyperparameters = {
    "DQN_Agents": {
        "linear_hidden_units": [10, 5],
        "learning_rate": 0.01,
        "buffer_size": 40000,
        "batch_size": 256,
        "final_layer_activation": "None",
        "columns_of_data_to_be_embedded": [0],
        "embedding_dimensions": [[config.environment.observation_space.n,
                                  max(4, int(config.environment.observation_space.n / 10.0))]],
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "update_every_n_steps": 1,
        "epsilon_decay_rate_denominator": 400,
        "discount_rate": 0.99,
        "learning_iterations": 1,
        "tau": 0.01
    },

    "SNN_HRL": {
        "SKILL_AGENT": {
            "num_skills": 20,
            "regularisation_weight": 1.5,
            "visitations_decay": 0.9999,
            "episodes_for_pretraining": 300,
            "batch_size": 256,
            "learning_rate": 0.001,
            "buffer_size": 40000,
            "linear_hidden_units": [20, 10],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0, 1],
            "embedding_dimensions": [[config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))],
                                     [20, 6]],
            "batch_norm": False,
            "gradient_clipping_norm": 2,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 500,
            "discount_rate": 0.999,
            "learning_iterations": 1,
            "tau": 0.01
        },

        "MANAGER": {
            "timesteps_before_changing_skill": 6,
            "linear_hidden_units": [10, 5],
            "learning_rate": 0.01,
            "buffer_size": 40000,
            "batch_size": 256,
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": [[config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))]],
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 50,
            "discount_rate": 0.99,
            "learning_iterations": 1,
            "tau": 0.01

        }

    }
}

if __name__== '__main__':

    print(config.environment.reward_for_achieving_goal)

    AGENTS = [SNN_HRL] #, DDQN]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()


