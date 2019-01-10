from Agents.DQN_Agents.DQN_Agent import DQN_Agent
from HER_Extension import HER_Extension

class DQN_HER_Agent(DQN_Agent, HER_Extension):
    """DQN algorithm with hindsight experience replay"""
    agent_name = "DQN_HER"

    def __init__(self, config):
        DQN_Agent.__init__(self, config)

    def step(self):
        """Runs a step within a game including a learning step if required"""
        self.pick_and_conduct_action()
        self.update_next_state_reward_done_and_score()
        if self.time_for_critic_to_learn():
            self.critic_learn()
        self.track_episodes_data()
        self.save_experience()

        if self.done:
            self.save_alternative_experience()

        self.state = self.next_state  # this is to set the state for the next iteration