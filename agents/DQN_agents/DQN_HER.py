from agents.DQN_agents.DQN import DQN
from agents.HER_Base import HER_Base

class DQN_HER(HER_Base, DQN):
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
            self.conduct_action_in_changeable_goal_envs(self.action)
            if self.time_for_q_network_to_learn():
                for _ in range(self.hyperparameters["learning_iterations"]):
                    self.learn(experiences=self.sample_from_HER_and_Ordinary_Buffer())
            self.track_changeable_goal_episodes_data()
            self.save_experience()
            if self.done: self.save_alternative_experience()
            self.state_dict = self.next_state_dict  # this is to set the state for the next iteration
            self.state = self.next_state
            self.global_step_number += 1
        self.episode_number += 1

    def enough_experiences_to_learn_from(self):
        """Returns booleans indicating whether there are enough experiences in the two replay buffers to learn from"""
        return len(self.memory) > self.ordinary_buffer_batch_size and len(self.HER_memory) > self.HER_buffer_batch_size