from Agents.DQN_Agents.DQN_HER import DQN_HER
from Environments.Four_Rooms_Environment import Four_Rooms_Environment
from Trainer import Trainer
from Utilities.Data_Structures.Config import Config
from Agents.DQN_Agents.DQN import DQN

config = Config()
config.seed = 1
config.environment = Four_Rooms_Environment(9, 9, stochastic_actions_probability=0.0, random_start_user_place=True, random_goal_place=True)
config.num_episodes_to_run = 4500
config.file_to_save_data_results = "Data_and_Graphs/Four_Rooms_Environment_Results_Data.pkl"
config.file_to_save_results_graph = "Data_and_Graphs/Four_Rooms_Environment_Results_Graph.png"
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
        "learning_rate": 0.001,
        "batch_size": 128,
        "buffer_size": 100000,
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 500,
        "discount_rate": 0.98,
        "tau": 0.1,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.4,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "linear_hidden_units": [10, 10],
        "columns_of_data_to_be_embedded": [0, 1],
        "embedding_dimensions": [[config.environment.get_num_possible_states(),
                                      max(4, int(config.environment.get_num_possible_states() / 10.0))] for _ in range(2)],
        "final_layer_activation": None,
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "HER_sample_proportion": 0.8
    }
}

if __name__== '__main__':
    AGENTS = [DQN_HER, DQN]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()


