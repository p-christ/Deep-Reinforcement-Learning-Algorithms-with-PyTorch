# NOT FINISHED
from Replay_Buffer import Replay_Buffer

class Memory_Shaper(object):
    """Takes in the experience of full episodes and reshapes it according to macro-actions you define. Then it provides
    a replay buffer with this reshaped data to learn from"""

    def __init__(self, buffer_size, batch_size, seed):
        self.reset(buffer_size, batch_size, seed)

    def add_adapted_experiences_to_replay_buffer(self, action_rules):
        """Adds experiences to the replay buffer after re-imagining that the actions taken were macro-actions according to
         action_rules as well as primitive actions.

         NOTE that we want to put both primitive actions and macro-actions into replay buffer so that it can learn that
         its better to do a macro-action rather than the same primitive actions (which we will enforce with reward penalty)
         """
        for key in action_rules.keys():
            assert isinstance(key, tuple)
            assert isinstance(action_rules[key], int)

        episodes = len(self.states)
        for data_type in [self.states, self.next_states, self.rewards, self.actions, self.dones]:
            assert len(data_type) == episodes

        for episode_ix in range(episodes):
            self.add_adapted_experience_for_an_episode(episode_ix)

    def add_adapted_experience_for_an_episode(self, episode_ix):

        states = self.states[episode_ix]
        next_states = self.next_states[episode_ix]
        rewards = self.rewards[episode_ix]
        actions = self.actions[episode_ix]
        dones = self.done[episode_ix]
        assert len(states) == len(next_states) == len(rewards) == len(dones) == len(actions)
        steps = len(states)

        for step in range(steps)

        pass

    def add_episode_experience(self, states, next_states, rewards, actions, dones):
        """Adds in an episode of experience"""
        self.states.append(states)
        self.next_states.append(next_states)
        self.rewards.append(rewards)
        self.actions.append(actions)
        self.dones.append(dones)

    def reset(self, buffer_size, batch_size, seed):
        self.states = []
        self.next_states = []
        self.rewards = []
        self.actions = []
        self.dones = []
        self.replay_buffer = Replay_Buffer(buffer_size, batch_size, seed)




