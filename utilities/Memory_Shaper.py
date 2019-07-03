# NOT FINISHED
from .data_structures.Action_Balanced_Replay_Buffer import Action_Balanced_Replay_Buffer
from .data_structures.Replay_Buffer import Replay_Buffer
import numpy as np
import random

class Memory_Shaper(object):
    """Takes in the experience of full episodes and reshapes it according to macro-actions you define. Then it provides
    a replay buffer with this reshaped data to learn from"""
    def __init__(self, buffer_size, batch_size, seed, new_reward_fn, action_balanced_replay_buffer=True):
        self.reset()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        self.new_reward_fn = new_reward_fn
        self.action_balanced_replay_buffer = action_balanced_replay_buffer

    def put_adapted_experiences_in_a_replay_buffer(self, action_id_to_actions):
        """Adds experiences to the replay buffer after re-imagining that the actions taken were macro-actions according to
         action_rules as well as primitive actions.

         NOTE that we want to put both primitive actions and macro-actions into replay buffer so that it can learn that
         its better to do a macro-action rather than the same primitive actions (which we will enforce with reward penalty)
         """

        actions_to_action_id = {v: k for k, v in action_id_to_actions.items()}

        self.num_actions = len(action_id_to_actions)

        print(actions_to_action_id)

        for key in actions_to_action_id.keys():
            assert isinstance(key, tuple)
            assert isinstance(actions_to_action_id[key], int)

        episodes = len(self.states)
        for data_type in [self.states, self.next_states, self.rewards, self.actions, self.dones]:
            assert len(data_type) == episodes

        max_action_length = self.calculate_max_action_length(actions_to_action_id)

        if self.action_balanced_replay_buffer:
            print("Using action balanced replay buffer")
            replay_buffer = Action_Balanced_Replay_Buffer(self.buffer_size, self.batch_size, self.seed, num_actions=self.num_actions)
        else:
            print("Using ordinary replay buffer")
            replay_buffer = Replay_Buffer(self.buffer_size, self.batch_size, self.seed)

        for episode_ix in range(episodes):
            self.add_adapted_experience_for_an_episode(episode_ix, actions_to_action_id, max_action_length, replay_buffer)

        return replay_buffer

    def calculate_max_action_length(self, actions_to_action_id):
        """Calculates the max length of the provided macro-actions"""
        max_length = 0
        for key in actions_to_action_id.keys():
            action_length = len(key)
            if action_length > max_length:
                max_length = action_length
        return max_length


    def add_adapted_experience_for_an_episode(self, episode_ix, action_rules, max_action_length, replay_buffer):
        """Adds all the experiences we have been given to a replay buffer after adapting experiences that involved doing a
          macro action"""
        states = self.states[episode_ix]
        next_states = self.next_states[episode_ix]
        rewards = self.rewards[episode_ix]
        actions = self.actions[episode_ix]
        dones = self.dones[episode_ix]

        assert len(states) == len(next_states) == len(rewards) == len(dones) == len(actions), "{} {} {} {} {} = {}".format(len(states), len(next_states), len(rewards), len(dones), len(actions), actions)
        steps = len(states)
        for step in range(steps):
            replay_buffer.add_experience(states[step], actions[step], rewards[step], next_states[step], dones[step])
            for action_length in range(2, max_action_length + 1):
                if step < action_length - 1: continue
                action_sequence =  tuple(actions[step - action_length + 1 : step + 1])
                assert all([action in range(self.num_actions) for action in action_sequence]), "All actions should be primitive here"
                if action_sequence in action_rules.keys():
                    new_action = action_rules[action_sequence]
                    new_state = states[step - action_length + 1]
                    new_reward = np.sum(rewards[step - action_length + 1:step + 1])
                    new_reward = self.new_reward_fn(new_reward, len(action_sequence))
                    new_next_state = next_states[step]
                    new_dones = dones[step]
                    replay_buffer.add_experience(new_state, new_action, new_reward, new_next_state, new_dones)


    def add_episode_experience(self, states, next_states, rewards, actions, dones):
        """Adds in an episode of experience"""
        self.states.append(states)
        self.next_states.append(next_states)
        self.rewards.append(rewards)
        self.actions.append(actions)
        self.dones.append(dones)

    def reset(self):
        self.states = []
        self.next_states = []
        self.rewards = []
        self.actions = []
        self.dones = []





