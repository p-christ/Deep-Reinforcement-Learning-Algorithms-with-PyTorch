# In
# our
# experiments, all  parameters
# are
# annealed
# from
# 1
# to
# 0.1
# over
# 50, 000
# steps.The
# learning
# rate is set
# to
# 0.00025


from A2C import A2C
from Agents.Policy_Gradient_Agents.PPO import PPO
from Trainer import Trainer
from Utilities.Data_Structures.Config import Config
from Agents.DQN_Agents.DDQN import DDQN
from Agents.DQN_Agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from Agents.DQN_Agents.DQN import DQN
from Agents.DQN_Agents.DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets
from Environments.Long_Corridor_Environment import Long_Corridor_Environment

config = Config()
config.seed = 1
config.environment = Long_Corridor_Environment()
config.num_episodes_to_run = 1500
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

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.00025,
        "batch_size": 128,
        "buffer_size": 40000,
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 1000,
        "discount_rate": 0.99,
        "tau": 0.1,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 3,
        "linear_hidden_units": [10, 10],
        "final_layer_activation": "None",
        "columns_of_data_to_be_embedded": [0],
        "embedding_dimensions": [[config.environment.get_num_possible_states(), max(4, int(config.environment.get_num_possible_states() / 10.0))]],
        "batch_norm": False,
        "gradient_clipping_norm": 5
    }
}

if __name__ == "__main__":
    AGENTS = [DQN]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()






