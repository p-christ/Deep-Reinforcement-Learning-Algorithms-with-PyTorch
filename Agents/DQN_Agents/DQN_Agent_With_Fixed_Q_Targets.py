import copy
from Agents.DQN_Agents.DQN_Agent import DQN_Agent

class DQN_Agent_With_Fixed_Q_Targets(DQN_Agent):
    """A DQN agent that uses an older version of the q_network as the target network"""
    agent_name = "DQN with Fixed Q Targets"

    def __init__(self, config):
        print("Initialising DQN_Agent_With_Fixed_Q_Targets Agent")
        DQN_Agent.__init__(self, config)
        self.q_network_target = copy.deepcopy(self.q_network_local).to(self.device)

    def q_network_learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        super(DQN_Agent_With_Fixed_Q_Targets, self).q_network_learn(experiences=experiences)
        self.soft_update_of_target_network(self.q_network_local, self.q_network_target,
                                           self.hyperparameters["tau"])  # Update the target network

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next