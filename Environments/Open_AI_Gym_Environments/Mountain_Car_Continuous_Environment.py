from Environments.Base_Environment import Base_Environment
import gym

class Mountain_Car_Continuous_Environment(Base_Environment):

    def __init__(self):
        self.game_environment = gym.make("MountainCarContinuous-v0")

        self.state = self.game_environment.reset()
        self.next_state = None
        self.reward = None
        self.done = False

    def conduct_action(self, action):
        self.next_state, self.reward, self.done, _ = self.game_environment.step(action)

    def get_action_size(self):
        return 1

    def get_state_size(self):
        return len(self.state)

    def get_state(self):
        return self.state

    def get_next_state(self):
        return self.next_state

    def get_reward(self):
        return self.reward

    def get_done(self):
        return self.done

    def reset_environment(self):
        self.state = self.game_environment.reset()

    def get_max_steps_per_episode(self):
        return 1000

    def get_action_types(self):
        return "CONTINUOUS"

    def get_score_to_win(self):
        return 90

    def get_rolling_period_to_calculate_score_over(self):
        return 100
