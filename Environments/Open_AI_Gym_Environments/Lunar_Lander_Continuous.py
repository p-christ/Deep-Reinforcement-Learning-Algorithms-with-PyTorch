import gym


class Lunar_Lander_Continuous(object):

    def __init__(self):
        self.game_environment = gym.make("LunarLanderContinuous-v2")
        self.state = self.game_environment.reset()
        self.next_state = None
        self.reward = None
        self.done = False
        gym.logger.set_level(40) #stops it from printing an unnecessary warning


    def conduct_action(self, action):
        self.next_state, self.reward, self.done, _ = self.game_environment.step(action)

    def get_action_size(self):
        return 2

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
        return float("inf")

    def get_action_types(self):
        return "CONTINUOUS"

    def get_score_to_win(self):
        return 200

    def get_rolling_period_to_calculate_score_over(self):
        return 100

