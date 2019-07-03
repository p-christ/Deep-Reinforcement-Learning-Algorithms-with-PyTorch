import random
from collections import namedtuple, deque
import torch
import numpy as np
from .Replay_Buffer import Replay_Buffer

class Action_Balanced_Replay_Buffer(Replay_Buffer):
    """Replay buffer that provides sample of experiences that have an equal number of each action being conducted"""
    def __init__(self, buffer_size, batch_size, seed, num_actions):
        self.num_actions = num_actions
        self.buffer_size_per_memory = int(buffer_size / self.num_actions)

        print("NUM ACTIONS ", self.num_actions)
        self.memories = {action: deque(maxlen=self.buffer_size_per_memory) for action in range(self.num_actions)}
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Adds experience or list of experiences into the replay buffer"""
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [self.experience(state, action, reward, next_state, done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]
            for experience in experiences:
                action = experience.action
                self.memories[action].append(experience)
        else:
            experience = self.experience(states, actions, rewards, next_states, dones)
            self.memories[actions].append(experience)

    def pick_experiences(self, num_experiences=None):
        """Picks the experiences that the sample function will return as a random sample of experiences. It works by picking
        an equal number of experiences that used each action (as far as possible)"""
        if num_experiences: batch_size = num_experiences
        else: batch_size = self.batch_size
        batch_per_action = self.calculate_batch_sizes_per_action(batch_size)
        samples_split_by_action = self.sample_each_action_equally(batch_per_action)
        combined_sample = []
        for key in samples_split_by_action.keys():
            combined_sample.extend(samples_split_by_action[key])
        return combined_sample

    def calculate_batch_sizes_per_action(self, batch_size):
        """Calculates the batch size we need to randomly draw from each action to make sure there is equal coverage
        per action and that the batch gets filled up"""
        min_batch_per_action = int(batch_size / self.num_actions)
        batch_per_action = {k: min_batch_per_action for k in range(self.num_actions)}
        current_batch_size = np.sum([batch_per_action[k] for k in range(self.num_actions)])
        remainder = batch_size - current_batch_size
        give_remainder_to = random.sample(range(self.num_actions), remainder)
        for action in give_remainder_to:
            batch_per_action[action] += 1
        return batch_per_action

    def sample_each_action_equally(self, batch_per_action):
        """Samples a number of experiences (determined by batch_per_action) from the memory buffer for each action"""
        samples = {}
        for action in range(self.num_actions):
            memory = self.memories[action]
            batch_size_for_action = batch_per_action[action]
            action_memory_size = len(memory)
            assert action_memory_size > 0, "Need at least 1 experience for each action"
            if action_memory_size >= batch_size_for_action:
                samples[action] = random.sample(memory, batch_size_for_action)
            else:
                print("Memory size {} vs. required batch size {}".format(action_memory_size, batch_size_for_action))
                samples_for_action = []
                while len(samples_for_action) < batch_per_action[action]:
                    remainder = batch_per_action[action] - len(samples_for_action)
                    sampled_experiences = random.sample(memory, min(remainder, action_memory_size))
                    samples_for_action.extend(sampled_experiences)
                samples[action] = samples_for_action
        return samples

    def __len__(self):
        return  np.sum([len(memory) for memory in self.memories.values()])

    def sample_experiences_with_certain_actions(self, allowed_actions, num_all_actions, required_batch_size):
        """Samples a number of experiences where the action conducted was in the list of required actions"""
        assert isinstance(allowed_actions, list)
        assert len(allowed_actions) > 0

        num_new_actions = len(allowed_actions)
        experiences_to_sample = int(required_batch_size * float(num_all_actions) / float(num_new_actions))
        experiences = self.sample(num_experiences=experiences_to_sample)
        states, actions, rewards, next_states, dones = experiences
        matching_indexes = np.argwhere((np.in1d(actions.numpy(), allowed_actions)))
        assert matching_indexes.shape[1] == 1

        matching_indexes = matching_indexes[:, 0]

        states = states[matching_indexes]
        actions = actions[matching_indexes]
        rewards = rewards[matching_indexes]
        next_states = next_states[matching_indexes]
        dones = dones[matching_indexes]

        assert abs(states.shape[0] - required_batch_size) <= 0.05*required_batch_size, "{} vs. {}".format(states.shape[0], required_batch_size)


        return (states, actions, rewards, next_states, dones)
