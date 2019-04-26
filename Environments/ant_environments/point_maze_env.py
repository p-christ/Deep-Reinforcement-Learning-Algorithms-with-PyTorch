


# NOTE THIS CODE IS TAKEN FROM https://github.com/tensorflow/models/tree/master/research/efficient-hrl/environments
# and is not my code.





from ant_environments.maze_env import MazeEnv
from ant_environments.point import PointEnv


class PointMazeEnv(MazeEnv):
    MODEL_CLASS = PointEnv
