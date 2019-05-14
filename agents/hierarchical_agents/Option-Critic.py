from Base_Agent import Base_Agent

# TBC

# initialise Q_U  --> Q values over option actions
# initialise Q_omega --> Q values over options
# initialise intra-option policy
#
#
# pick option using epsilon greedy policy
# play
#
# gradient update parameters of intra-option policy
# gradient update parameters of termination function

# class Option_Critic(Base_Agent):
#     """Agent from paper Option-Critic (Bacon et al. 2016) https://arxiv.org/pdf/1609.05140.pdf"""
#     agent_name = "Option Critic"
#
#     def __init__(self, config):
#         Base_Agent.__init__(self, config)
#
#         self.num_options = self.hyperparameters["num_options"]
#
#         self.q_options = self.create_NN(input_dim=self.state_size, output_dim=self.num_options, key_to_use="q_options")
#         self.q_option_actions = self.create_NN(input_dim=self.state_size + 1, output_dim=self.action_size, key_to_use="q_option_actions")
#
#
#
#         self.intra_option_policies = [self.create_NN(input_dim=self.state_size, output_dim=self.action_size) for _ in range(self.num_options)]
#
#
#
#
# #
#
#

