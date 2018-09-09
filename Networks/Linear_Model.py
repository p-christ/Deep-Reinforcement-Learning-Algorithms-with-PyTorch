
import numpy as np

class Linear_Model(object):
    def __init__(self, state_size=4, action_size=2):
        self.weights = 1e-4 * np.random.rand(state_size, action_size)  # weights for simple linear policy: state_space x action_space

    def forward(self, state):
        x = np.dot(state, self.weights)
        return np.exp(x) / sum(np.exp(x))

    def act(self, state):
        probs = self.forward(state)
        # action = np.random.choice(2, p=probs) # option 1: stochastic policy
        action = np.argmax(probs)  # option 2: deterministic policy
        return action