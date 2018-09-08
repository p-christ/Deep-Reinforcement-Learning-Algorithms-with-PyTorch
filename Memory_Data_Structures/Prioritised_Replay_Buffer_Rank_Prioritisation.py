import heapq as pq
from collections import namedtuple
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

class Prioritised_Replay_Buffer_Rank_Prioritisation(object):

    def __init__(self, max_memory_size, batch_size):

        self.memory = []
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])

        self.batch_size = batch_size
        self.max_memory_size = max_memory_size
        self.current_memory_size = 0


    def add(self, state, action, reward, next_state, done, td_error):

        experience = self.experience(state,
                                     action,
                                     reward,
                                     next_state,
                                     done)

        adapted_td_error = 1.0 / (abs(td_error) + 0.000001)

        if self.spare_space_in_memory():
            self.current_memory_size += 1

        else:
            self.memory.pop()

        try:

            pq.heappush(self.memory, (adapted_td_error, experience))

        except (RuntimeError, ValueError):

            print((adapted_td_error, experience))

    def sample(self):
        """Removes and returns batch_size amount of experiences with the highest td_errors. Note that this
        is not the same as in the DeepMind paper where they use stratified sampling instead."""

        sample = self.memory[:self.batch_size]
        sample_of_experiences = [values[1] for values in sample]
        states, actions, rewards, next_states, dones = self.separate_out_data_types(sample_of_experiences)

        self.remove_sampled_experiences()

        return states, actions, rewards, next_states, dones



    def separate_out_data_types(self, experiences):

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(device)

        return states, actions, rewards, next_states, dones

    def add_batch(self, states, actions, rewards, next_states, dones, td_errors):

        for state, action, reward, next_state, done, td_error in zip(states, actions, rewards, next_states, dones, td_errors):
            self.add(state.data.cpu.numpy(), action.data.cpu.numpy()[0],
                     reward.data.cpu.numpy()[0], next_state.data.cpu.numpy(),
                     done.data.cpu.numpy()[0], td_error)


    def spare_space_in_memory(self):

        return self.current_memory_size < self.max_memory_size

    def remove_sampled_experiences(self):

        self.memory = self.memory[self.batch_size:]
        self.current_memory_size -= self.batch_size

    def __len__(self):
        return len(self.memory)
