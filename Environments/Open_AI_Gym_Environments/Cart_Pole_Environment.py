import gym
import numpy as np
from pyvirtualdisplay import Display
import matplotlib.pyplot as plt

from Environments.Base_Environment import Base_Environment

class Cart_Pole_Environment(Base_Environment):

    def __init__(self):
        self.game_environment = gym.make("CartPole-v0")
        self.state = self.game_environment.reset()
        self.next_state = None
        self.reward = None
        self.done = False
        gym.logger.set_level(40) #stops it from printing an unnecessary warning

    def conduct_action(self, action):
        if type(action) is np.ndarray:
            action = action[0]
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

    def reset_environment(self):
        self.state = self.game_environment.reset()

    def visualise_agent(self, agent):

        env = self.game_environment

        display = Display(visible=0, size=(1400, 900))
        display.start()

        state = env.reset()
        img = plt.imshow(env.render(mode='rgb_array'))
        for t in range(1000):
            agent.step()
            img.set_data(env.render(mode='rgb_array'))
            plt.axis('off')
            display.display(env.gcf())
            display.clear_output(wait=True)
            if agent.done:
                break
        env.close()

    def get_max_steps_per_episode(self):
        return 200

    def get_action_types(self):
        return "DISCRETE"

    def get_score_to_win(self):
        return 195

    def get_rolling_period_to_calculate_score_over(self):
        return 100

