from Utilities.Models.Neural_Network import Neural_Network
from Agents.DQN_Agents.DQN_Agent import DQN_Agent


class DQN_Agent_With_Fixed_Q_Targets(DQN_Agent):
    agent_name = "DQN with Fixed Q Targets"

    def __init__(self, config):
        print("Initialising DQN_Agent_With_Fixed_Q_Targets Agent")
        DQN_Agent.__init__(self, config)
        self.q_network_target = Neural_Network(self.state_size, self.action_size, config.seed, self.hyperparameters, "VANILLA_NN").to(self.device)

    def q_network_learn(self, experiences_given=False, experiences=None):
        super(DQN_Agent_With_Fixed_Q_Targets, self).q_network_learn(experiences_given=experiences_given, experiences=experiences)

        if self.episode_number % 5 == 0:
            self.soft_update_of_target_network(self.q_network_local, self.q_network_target, 1.0) #Update the target network

        # self.soft_update_of_target_network(self.q_network_local, self.q_network_target,
        #                                    self.hyperparameters["tau"])  # Update the target network

    def compute_q_values_for_next_states(self, next_states):
        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next
