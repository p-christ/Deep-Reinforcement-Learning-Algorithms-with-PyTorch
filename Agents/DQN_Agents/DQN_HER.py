from Agents.DQN_Agents.DQN import DQN
from Agents.HER_Base import HER_Base

class DQN_HER(DQN, HER_Base):
    """DQN algorithm with hindsight experience replay"""
    agent_name = "DQN-HER"

    def __init__(self, config):
        DQN.__init__(self, config)
        HER_Base.__init__(self, self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"],
                          self.hyperparameters["HER_sample_proportion"])

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.action = self.pick_action()
            self.conduct_action(self.action)
            if self.time_for_q_network_to_learn():
                self.learn(experiences=self.sample_from_HER_and_Ordinary_Buffer())
            self.track_episodes_data()
            self.save_experience()
            if self.done: self.save_alternative_experience()
            self.state = self.next_state  # this is to set the state for the next iteration
            self.global_step_number += 1
        self.episode_number += 1

    def enough_experiences_to_learn_from(self):
        """Returns booleans indicating whether there are enough experiences in the two replay buffers to learn from"""
        return len(self.memory) > self.ordinary_buffer_batch_size and len(self.HER_memory) > self.HER_buffer_batch_size