import copy
import random
import numpy as np
from Agents.Base_Agent import Base_Agent
from Agents.Policy_Gradient_Agents.PPO import PPO


class SNNHRL(Base_Agent):
    """Implements the hierarchical RL agent that uses stochastic neural networks (SNN) from the paper Florensa et al. 2017
    https://arxiv.org/pdf/1704.03012.pdf

    Works by:
    1) Creating a pre-training environment within which the skill_agent can learn for some period of time
    2) Then skill_agent is frozen
    3) Then we train a manager agent that chooses which of the pre-trained skills to let act for it for some period of time

    Note that it only works with discrete states at the moment.
    """
    agent_name = "SNNHRL"

    def __init__(self, config):
        Base_Agent.__init__(self, config)

        self.num_skills = self.hyperparameters["SKILL_AGENT"]["num_skills"]

        self.skill_agent_config = copy.deepcopy(config)
        self.skill_agent_config.hyperparameters = self.skill_agent_config.hyperparameters["SKILL_AGENT"]
        # self.skill_agent_config.state_size = self.state_size + 1
        # self.skill_agent = PPO(self.skill_agent_config)

        self.skill = 10

        self.env_parameters = config.env_parameters




        self.create_skill_learning_environment()

        assert isinstance(self.environment.state, int), "only works for discrete states currently"







        self.episodes_for_pretraining = 100

    def create_skill_learning_environment(self):

        meta_agent = self
        environment_class = self.environment.__class__


        print("Environment class ", environment_class)

        class skills_env(environment_class):

            def __init__(self, meta_agent):
                environment_class.__init__(self, **meta_agent.env_parameters)
                self.meta_agent = meta_agent
                self.state_visitations = [[0 for _ in range(self.get_num_possible_states())] for _ in
                                          range(self.meta_agent.num_skills)]

                print(self.meta_agent.skill)

            def get_state_size(self):
                return super().get_state_size() + 1

            def get_state(self):
                return np.concatenate((super().get_state(), np.array([self.meta_agent.skill])))

            def get_next_state(self):
                return np.concatenate((super().get_next_state(), np.array([self.meta_agent.skill])))
            #
            # def conduct_action(self, action):
            #     super().conduct_action(action)
            #     self.update_state_visitations()
            #     self.update_reward_with_mutual_information_criterion()
            #
            # def update_state_visitations(self):
            #     self.state_visitations[self.meta_agent.skill][super().get_state()] += 1

                # update visitations count...
                # add to reward



        self.skill_agent_config.environment = skills_env(meta_agent)
        # agent.
        #
        # new_skills_env = skills_env()
        #
        #
        #
        # environment = self.skill_agent_config.environment
        #
        # print(environment.get_state_size())
        #
        #
        #
        # # state size...
        # environment.get_state_size = lambda : self.environment.get_state_size() + 1
        #
        # print(environment.get_state_size())
        #
        # print(environment.get_state())
        #
        #
        # print(np.array(environment.state))
        # print(np.array([self.skill]))
        #
        # environment.get_state = lambda : np.concatenate((np.array([environment.state]), np.array([self.skill])))
        #
        # print(environment.get_state())
        #
        # environment.get_next_state = lambda : np.concatenate((self.environment.get_next_state(), np.array([skill])))
        #



    # def step(self):
    #
    #
    #     if self.episode_number <= self.episodes_for_pretraining:
    #         self.skill_training_step()
    #
    #     else:
    #         self.post_training_step()
    #
    # def skill_training_step(self):
    #     """Runs a step of the game in skill training stage"""
    #
    #     randomly_chosen_skill = random.randint(0, self.num_skills - 1)
    #
    #     while not self.done:
    #
    #         # change the environment... keep agent the same...
    #
    #
    #
    #     # play episode with z fixed
    #
    #     # calculate rewards
    #     # learn SNN through PPO learning
    #
    #     self.pre_training()
    #
    #     # then SNN learn... using PPO learning regime
    #
    # def post_training_step(self):
    #     """Runs a step of the game after we have frozen the SNN and are just training the manager"""
    #
    #     # manager takes in state and selects skill probabilities
    #     # latent code sampled according to skill probabilities and fed into SNN which then acts in world for T steps
    #     # manager uses PPO to learn

#
# from Trainer import Trainer
# from Utilities.Data_Structures.Config import Config
# from Agents.DQN_Agents.DQN import DQN
# from Agents.Hierarchical_Agents.h_DQN import h_DQN
# from Environments.Long_Corridor_Environment import Long_Corridor_Environment
# config = Config()
# config.seed = 1
# config.environment = Long_Corridor_Environment(stochasticity_of_action_right=0.5)
# config.num_episodes_to_run = 10000
# config.file_to_save_data_results = None
# config.file_to_save_results_graph = None
# config.show_solution_score = False
# config.visualise_individual_results = False
# config.visualise_overall_agent_results = True
# config.standard_deviation_results = 1.0
# config.runs_per_agent = 3
# config.use_GPU = False
# config.overwrite_existing_results_file = False
# config.randomise_random_seed = True
# config.save_model = False
#
# config.hyperparameters = {
#
#     "SNNHRL": {
#         "SKILL_AGENT": {
#             "num_skills": 10,
#             "batch_size": 256,
#             "learning_rate": 0.01,
#             "buffer_size": 40000,
#             "linear_hidden_units": [20, 10],
#             "final_layer_activation": "None",
#             "columns_of_data_to_be_embedded": [0, 1],
#             "embedding_dimensions": [[config.environment.get_num_possible_states(),
#                                       max(4, int(config.environment.get_num_possible_states() / 10.0))],
#                                      [config.environment.get_num_possible_states(),
#                                       max(4, int(config.environment.get_num_possible_states() / 10.0))]],
#             "batch_norm": False,
#             "gradient_clipping_norm": 5,
#             "update_every_n_steps": 1,
#             "epsilon_decay_rate_denominator": 1500,
#             "discount_rate": 0.999,
#             "learning_iterations": 1
#         },
#         "META_CONTROLLER": {
#             "batch_size": 256,
#             "learning_rate": 0.001,
#             "buffer_size": 40000,
#             "linear_hidden_units": [20, 10],
#             "final_layer_activation": "None",
#             "columns_of_data_to_be_embedded": [0],
#             "embedding_dimensions": [[config.environment.get_num_possible_states(),
#                                       max(4, int(config.environment.get_num_possible_states() / 10.0))]],
#             "batch_norm": False,
#             "gradient_clipping_norm": 5,
#             "update_every_n_steps": 1,
#             "epsilon_decay_rate_denominator": 2500,
#             "discount_rate": 0.999,
#             "learning_iterations": 1
#         }
#     }
# }
#
# config.hyperparameters = config.hyperparameters["SNNHRL"]
#
# agent = SNNHRL(config)