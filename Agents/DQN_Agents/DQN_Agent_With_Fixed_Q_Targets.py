import copy
from Agents.DQN_Agents.DQN_Agent import DQN_Agent


class DQN_Agent_With_Fixed_Q_Targets(DQN_Agent):
    agent_name = "DQN with Fixed Q Targets"

    def __init__(self, config):
        print("Initialising DQN_Agent_With_Fixed_Q_Targets Agent")
        DQN_Agent.__init__(self, config)
        self.q_network_target = copy.deepcopy(self.q_network_local).to(self.device)

    def q_network_learn(self, experiences=None):
        super(DQN_Agent_With_Fixed_Q_Targets, self).q_network_learn(experiences=experiences)
        self.soft_update_of_target_network(self.q_network_local, self.q_network_target,
                                           self.hyperparameters["tau"])  # Update the target network

    def compute_q_values_for_next_states(self, next_states):
        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next
