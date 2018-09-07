from DQN_Agents.DDQN_Agent import DDQN_Agent


class Duelling_DQN(DDQN_Agent):

    def __init__(self, environment, seed, hyperparameters, rolling_score_length,
                 average_score_required, agent_name):
        DDQN_Agent.__init__(self, environment=environment,
                            seed=seed, hyperparameters=hyperparameters, rolling_score_length=rolling_score_length,
                            average_score_required=average_score_required, agent_name=agent_name)

        self.qnetwork_local = Q_Network(self.state_size, self.action_size, seed, hyperparameters).to(self.device)

        self.qnetwork_target = Q_Network(self.state_size, self.action_size, seed, hyperparameters).to(self.device)

