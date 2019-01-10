from DDPG_Agent import DDPG_Agent
from HER_Extension import HER_Extension

class DDPG_HER_Agent(DDPG_Agent, HER_Extension):
    """DDPG algorithm with hindsight experience replay"""
    agent_name = "DDPG_HER"

    def __init__(self, config):
        DDPG_Agent.__init__(self, config)

    def step(self):
        """Runs a step within a game including a learning step if required"""
        self.pick_and_conduct_action()
        self.update_next_state_reward_done_and_score()

        if self.time_for_critic_and_actor_to_learn():
            for _ in range(self.ddpg_hyperparameters["learning_updates_per_learning_session"]):
                states, actions, rewards, next_states, dones = self.sample_experiences()  # Sample experiences
                self.critic_learn(experiences_given=True, experiences=(states, actions, rewards, next_states, dones))
                self.actor_learn(states)
        self.track_episodes_data()
        self.save_experience()

        if self.done:
            self.save_alternative_experience()

        self.state = self.next_state #this is to set the state for the next iteration

