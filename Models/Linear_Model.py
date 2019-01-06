
import numpy as np

class Linear_Model(object):
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.state_size = state_size
        self.weights = 1e-4 * np.random.rand(self.state_size, self.action_size)  # weights for simple linear policy: state_space x action_space

    def forward(self, state):
        x = np.dot(state, self.weights)
        return np.exp(x) / sum(np.exp(x))

