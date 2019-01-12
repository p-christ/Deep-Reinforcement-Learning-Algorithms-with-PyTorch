from DDPG_Agent import DDPG_Agent
from HER_Base import HER_Base

class DDPG_HER_Agent(DDPG_Agent, HER_Base):
    """DDPG algorithm with hindsight experience replay"""
    agent_name = "DDPG_HER"

    def __init__(self, config):
        DDPG_Agent.__init__(self, config)

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.pick_and_conduct_action()
            self.update_next_state_reward_done_and_score()
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    states, actions, rewards, next_states, dones = self.memory.sample()  # Sample experiences
                    self.critic_learn(states, actions, rewards, next_states, dones)
                    self.actor_learn(states)
            self.track_episodes_data()
            self.save_experience()
            if self.done:
                self.save_alternative_experience()
            self.state = self.next_state #this is to set the state for the next iteration
            self.episode_step_number += 1
        self.episode_number += 1