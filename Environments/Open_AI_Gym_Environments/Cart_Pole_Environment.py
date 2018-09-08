from Environments.Base_Environment import Base_Environment
import gym

class Cart_Pole_Environment(Base_Environment):

    def __init__(self):
        self.game_environment = gym.make('CartPole-v0')

        self.state = self.game_environment.reset()
        self.next_state = None
        self.reward = None
        self.done = False

    def conduct_action(self, action):
        self.next_state, self.reward, self.done, _ = self.game_environment.step(action)

    def get_action_size(self):
        return self.game_environment.action_space.n

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

    def get_reward(self):
        return self.reward

    def reset_environment(self):
        self.state = self.game_environment.reset()