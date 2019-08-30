import torch
from torch import optim
from agents.Base_Agent import Base_Agent
from agents.DQN_agents.DDQN import DDQN

class Dueling_DDQN(DDQN):
    """A dueling double DQN agent as described in the paper http://proceedings.mlr.press/v48/wangf16.pdf"""
    agent_name = "Dueling DDQN"

    def __init__(self, config):
        DDQN.__init__(self, config)
        self.q_network_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size + 1)
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(), lr=self.hyperparameters["learning_rate"], eps=1e-4)
        self.q_network_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size + 1)
        Base_Agent.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)

    def pick_action(self, state=None):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if state is None: state = self.state
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
            action_values = action_values[:, :-1] #because we treat the last output element as state-value and rest as advantages
        self.q_network_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})
        return action

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network. Double DQN
        uses the local index to pick the maximum q_value action and then the target network to calculate the q_value.
        The reasoning behind this is that it will help stop the network from overestimating q values"""
        max_action_indexes = self.q_network_local(next_states)[:, :-1].detach().argmax(1)
        duelling_network_output = self.q_network_target(next_states)
        q_values = self.calculate_duelling_q_values(duelling_network_output)
        Q_targets_next = q_values.gather(1, max_action_indexes.unsqueeze(1))
        return Q_targets_next

    def calculate_duelling_q_values(self, duelling_q_network_output):
        """Calculates the q_values using the duelling network architecture. This is equation (9) in the paper
        referenced at the top of the class"""
        state_value = duelling_q_network_output[:, -1]
        avg_advantage = torch.mean(duelling_q_network_output[:, :-1], dim=1)
        q_values = state_value.unsqueeze(1) + (duelling_q_network_output[:, :-1] - avg_advantage.unsqueeze(1))
        return q_values

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        duelling_network_output = self.q_network_local(states)
        q_values = self.calculate_duelling_q_values(duelling_network_output)
        Q_expected = q_values.gather(1, actions.long())
        return Q_expected







