from gym.wrappers import FlattenDictWrapper
from agents.DQN_agents.DQN_HER import DQN_HER
from environments.Bit_Flipping_Environment import Bit_Flipping_Environment
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from agents.DQN_agents.DQN import DQN

config = Config()
config.seed = 1
config.environment = Bit_Flipping_Environment(14)
config.num_episodes_to_run = 4500
config.file_to_save_data_results = None #"Data_and_Graphs/Bit_Flipping_Results_Data.pkl"
config.file_to_save_results_graph = None #"Data_and_Graphs/Bit_Flipping_Results_Graph.png"
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
        "epsilon_decay_rate_denominator": 150,
        "discount_rate": 0.999,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "linear_hidden_units": [64, 64],
        "final_layer_activation": None,
        "y_range": (-1, 14),
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "HER_sample_proportion": 0.8,
        "learning_iterations": 1,
        "clip_rewards": False
    }
}

if __name__== '__main__':
    AGENTS = [DQN_HER, DQN]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()


