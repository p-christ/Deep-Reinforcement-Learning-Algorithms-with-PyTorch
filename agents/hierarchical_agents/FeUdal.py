from Base_Agent import Base_Agent

# # TODO add perception module
# # TODO add parts of network in paper not implemented
#
# # TBC
#
# class FeUdal(Base_Agent):
#     """Agent from paper FeUdal Networks for Hierarchical Reinforcement Learning (Vezhevets et al. 2017)
#     https://arxiv.org/pdf/1703.01161.pdf. Note that we have tried to use the same notation as used in the paper."""
#     agent_name = "FeUdal"
#
#     def __init__(self, config):
#         Base_Agent.__init__(self, config)
#
#         self.goal_dimension = self.hyperparameters["goal_dimension"]
#         self.manager_choice_frequency = self.hyperparameters["manager_choice_frequency"]
#
#         self.M_goal_network = self.create_NN(input_dim=self.state_size, output_dim=self.goal_dimension)
#         self.W_U_network = self.create_NN(input_dim=self.state_size, output_dim=self.action_size * self.goal_dimension)
#         self.W_goal_embedding = self.create_NN(input_dim=self.goal_dimension, output_dim=self.goal_dimension)

    # def step(self):
    #     """Runs a step within a game including a learning step if required"""
    #     while not self.done:
    #
    #         if self.time_M_pick_goal():
    #             self.M_pick_goal()
    #
    #         self.W_pick_conduct_action()
    #
    #         self.update_next_state_reward_done_and_score()
    #
    #
    #         self.pick_and_conduct_action()
    #
    #         if self.time():
    #             self.q_network_learn()
    #         self.save_experience()
    #         self.state = self.next_state #this is to set the state for the next iteration
    #         self.global_step_number += 1
    #     self.episode_number += 1
    #
    #




    # higher level agent produces goal





