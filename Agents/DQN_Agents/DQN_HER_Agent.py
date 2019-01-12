from Agents.DQN_Agents.DQN_Agent import DQN_Agent
from Agents.HER_Base import HER_Base

class DQN_HER_Agent(DQN_Agent, HER_Base):
    """DQN algorithm with hindsight experience replay"""
    agent_name = "DQN_HER"

    def __init__(self, config):
        DQN_Agent.__init__(self, config)

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.pick_and_conduct_action()
            self.update_next_state_reward_done_and_score()
            if self.time_for_q_network_to_learn():
                self.q_network_learn()
            self.track_episodes_data()
            self.save_experience()

            if self.done:
                self.save_alternative_experience()

            self.state = self.next_state  # this is to set the state for the next iteration
            self.episode_step_number += 1
        self.episode_number += 1