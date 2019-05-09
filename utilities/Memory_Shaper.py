
# NOT FINISHED

class Memory_Shaper(object):
    """Takes in the experience of full episodes and reshapes it according to macro-actions you define. Then it provides
    a replay buffer with this reshaped data to learn from"""

    def __init__(self):

        self.states = []
        self.next_states = []
        self.rewards = []
        self.actions = []
        self.dones = []

        pass


    def add_episode_experience(self, states, next_states, rewards, actions, dones):
        """Adds in an episode of experience"""
        self.states.append(states)
        self.next_states.append(next_states)
        self.rewards.append(rewards)
        self.actions.append(actions)
        self.dones.append(dones)




