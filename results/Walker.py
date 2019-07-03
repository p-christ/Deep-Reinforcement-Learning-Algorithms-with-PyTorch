import gym
from agents.policy_gradient_agents.PPO import PPO
from agents.actor_critic_agents.DDPG import DDPG
from agents.actor_critic_agents.SAC import SAC
from agents.actor_critic_agents.TD3 import TD3
from agents.Trainer import Trainer
from agents.hierarchical_agents.DIAYN import DIAYN
from utilities.data_structures.Config import Config


config = Config()
config.seed = 1
config.environment = gym.make("Walker2d-v2")
config.num_episodes_to_run = 400
config.file_to_save_data_results = "data_and_graphs/Walker_Results_Data.pkl"
config.file_to_save_results_graph = "data_and_graphs/Walker_Results_Graph.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 3
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False


actor_critic_agent_hyperparameters = {
        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 400,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True,
        "clip_rewards": clip_rewards
    }

dqn_agent_hyperparameters =   {
        "learning_rate": 0.005,
        "batch_size": 128,
        "buffer_size": 40000,
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 3,
        "discount_rate": 0.99,
        "tau": 0.01,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 3,
        "linear_hidden_units": [30, 15],
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "clip_rewards": clip_rewards
    }


manager_hyperparameters = dqn_agent_hyperparameters
manager_hyperparameters.update({"timesteps_to_give_up_control_for": 5})


config.hyperparameters = {
    "Policy_Gradient_Agents": {
            "learning_rate": 0.05,
            "linear_hidden_units": [30, 15],
            "final_layer_activation": "TANH",
            "learning_iterations_per_round": 10,
            "discount_rate": 0.9,
            "batch_norm": False,
            "clip_epsilon": 0.2,
            "episodes_per_learning_round": 10,
            "normalise_rewards": True,
            "gradient_clipping_norm": 5,
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.2,
            "epsilon_decay_rate_denominator": 1,
            "clip_rewards": clip_rewards
        },

    "Actor_Critic_Agents": actor_critic_agent_hyperparameters,
    "DIAYN": {
        "DISCRIMINATOR": {
            "learning_rate": 0.001,
            "linear_hidden_units": [32, 32],
            "final_layer_activation": None,
            "gradient_clipping_norm": 5

        },
        "AGENT": actor_critic_agent_hyperparameters,
        "MANAGER": manager_hyperparameters,
        "num_skills": 10,
        "num_unsupservised_episodes": 100
    }
}


if __name__ == "__main__":
    AGENTS = [DIAYN] #, SAC] #, DDPG, PPO, TD3] ] #,DIAYN] #
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()






