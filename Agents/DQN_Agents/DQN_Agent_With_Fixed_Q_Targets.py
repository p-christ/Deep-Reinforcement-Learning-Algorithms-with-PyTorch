from Neural_Network import Neural_Network
from Agents.DQN_Agents.DQN_Agent import DQN_Agent


class DQN_Agent_With_Fixed_Q_Targets(DQN_Agent):
    agent_name = "DQN with Fixed Q Targets"

    def __init__(self, config):
        print("Initialising DQN_Agent_With_Fixed_Q_Targets Agent")
        DQN_Agent.__init__(self, config)
        self.critic_target = Neural_Network(self.state_size, self.action_size, config.seed, self.hyperparameters).to(self.device)

    def critic_learn(self, experiences_given=False, experiences=None):
        super(DQN_Agent_With_Fixed_Q_Targets, self).critic_learn(experiences_given=experiences_given, experiences=experiences)
        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.hyperparameters["tau"]) #Update the target network

    def compute_q_values_for_next_states(self, next_states):
        Q_targets_next = self.critic_target(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next
    
    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size 
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
