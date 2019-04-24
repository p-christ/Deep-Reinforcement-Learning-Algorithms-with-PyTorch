from ant_environments.create_maze_env import create_maze_env

"""Environments taken from HIRO paper github repo: https://github.com/tensorflow/models/tree/master/research/efficient-hrl"""

class Ant_Navigation_Environments():

    def __init__(self, environment_name):
        self.env = create_maze_env(env_name=environment_name).gym  # names ["AntGather" ,"AntMaze", "AntPush", "AntFall", "AntBlock"]
        self.unwrapped = self.env.unwrapped
        self.spec = self.env.spec
        self.action_space = self.env.action_space

    def reset(self):
        self.steps_taken = 0
        return self.env.reset()

    def step(self, action):
        self.steps_taken += 1
        next_state, reward, _, _ =  self.env.step(action)
        if self.steps_taken >= 500: done = True
        else: done = False
        return next_state, reward, done, _



