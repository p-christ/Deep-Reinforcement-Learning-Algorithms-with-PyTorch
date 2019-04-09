from Actor_Critic_Agents.DDPG_Agent import DDPG_Agent
from DDPG_HER_Agent import DDPG_HER_Agent
from Data_Structures.Config import Config
from Fetch_Reach_Environment import Fetch_Reach_Environment
from Trainer import Trainer


config = Config()
config.seed = 1
config.environment = Fetch_Reach_Environment()
config.num_episodes_to_run = 2500
config.file_to_save_data_results = "Data_and_Graphs/Fetch_Reach_Results_Data.pkl"
config.file_to_save_results_graph = "Data_and_Graphs/Fetch_Reach_Results_Graph.png"
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
        "sigma": 0.25,  # 0.22 did well before
        "update_every_n_steps": 10,
        "learning_updates_per_learning_session": 10
    }
}


if __name__== '__main__':
    AGENTS = [DDPG_HER_Agent]#, DDPG_Agent]

    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
    #
    # trainer.visualise_preexisting_results(config.file_to_save_results_graph)


