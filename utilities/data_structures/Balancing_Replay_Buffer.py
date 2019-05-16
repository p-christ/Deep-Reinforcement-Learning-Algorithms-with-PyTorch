import random
from collections import namedtuple, deque
import torch
import numpy as np
from Replay_Buffer import Replay_Buffer

class Action_Balanced_Replay_Buffer(Replay_Buffer):
    """Replay buffer that provides sample of experiences that have an equal number of each action being conducted"""
    def __init__(self, buffer_size, batch_size, seed, num_actions):
        self.num_actions = num_actions
        self.buffer_size_per_memory = int(buffer_size / self.num_actions)
        self.memories = {action: deque(maxlen=self.buffer_size_per_memory) for action in range(self.num_actions)}
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Adds experience(s) into the replay buffer"""
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
        if num_experiences: batch_size = num_experiences
        else: batch_size = self.batch_size


        batch_per_action = self.batch_size / self.num_actions

        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return  np.sum([len(memory) for memory in self.memories.values()])
