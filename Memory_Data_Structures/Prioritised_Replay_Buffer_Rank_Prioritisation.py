import heapq as pq
import random
from collections import namedtuple
import numpy as np
import torch
from operator import itemgetter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

# use https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

class Prioritised_Replay_Buffer_Rank_Prioritisation(object):

    def __init__(self, max_memory_size, batch_size):

        self.memory = []
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])

        self.batch_size = batch_size
        self.max_memory_size = max_memory_size
        self.current_memory_size = 0
        self.counter = 0


    def add(self, state, action, reward, next_state, done, td_error):

        experience = self.experience(state,
                                     action,
                                     reward,
                                     next_state,
                                     done)

        # adapted_td_error = 1.0 / (abs(td_error) + 0.000001)
        adapted_td_error = abs(td_error)

                           # + random.random() / 100000000000000.0

        if self.spare_space_in_memory():
            try:
                pq.heappush(self.memory, (adapted_td_error, experience))
                self.current_memory_size += 1
            except (RuntimeError, ValueError) as e:
                # This error happens when the td_error for two entries is identical and the
                # heap doesn't know how to compare the entries properly
                pass

        else:
            try:
                pq.heappush(self.memory, (adapted_td_error, experience))
                pq.heappop(self.memory)
                # pq.heapreplace(self.memory, (adapted_td_error, experience))
            except (RuntimeError, ValueError) as e:
                # This error happens when the td_error for two entries is identical and the
                # heap doesn't know how to compare the entries properly
                pass


        #     self.memory.pop()
        #
        # try:
        #
        #     pq.heappush(self.memory, (adapted_td_error, experience))
        #
        # except (RuntimeError, ValueError):
        #
        #     print((adapted_td_error, experience))

    def sample(self):
        """Removes and returns batch_size amount of experiences with the highest td_errors. Note that this
        is not the same as in the DeepMind paper where they use stratified sampling instead."""

        sample = self.memory[-self.batch_size: ]
        sample_of_experiences = [values[1] for values in sample]
        states, actions, rewards, next_states, dones = self.separate_out_data_types(sample_of_experiences)

        self.remove_sampled_experiences()


        self.counter += 1

        if self.counter % 1000 == 0:
            print("\n" , self.current_memory_size)

        return states, actions, rewards, next_states, dones
    #
    # def sample(self):
    #
    #     total= sum(range(self.current_memory_size))
    #
    #     probabilities = [ix / total for ix in range(self.current_memory_size)]
    #
    #
    #     print(self.current_memory_size)
    #
    #     ix_choices = np.random.choice(range(self.current_memory_size), self.batch_size, p=probabilities)
    #     sample = itemgetter(*ix_choices)(self.memory)
    #     sample_of_experiences = [values[1] for values in sample]
    #     states, actions, rewards, next_states, dones = self.separate_out_data_types(sample_of_experiences)
    #
    #     self.memory = [x for ix, x in enumerate(self.memory) if ix not in set(ix_choices)]
    #     self.current_memory_size -= self.batch_size
    #
    #     return states, actions, rewards, next_states, dones

    # def rand_choice(self, start, end):
    #     return np.random.choice(range(start, end))
    #
    # def stratified_sample(self):
    #
    #     index_choices = []
    #
    #     start = 0
    #     end = 1
    #
    #     while len(index_choices) < self.batch_size:
    #
    #         index_choices.append(self.rand_choice(start, end))
    #
    #         start = end
    #         end = end * 2
    #
    #         if end > self.current_memory_size:
    #             start = 0
    #             end = 1
    #
    #     return index_choices
    # #

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

            self.add(state.data.numpy(), action.data.numpy()[0],
                     reward.data.numpy()[0], next_state.data.numpy(),
                     done.data.numpy()[0], float(td_error))


    def spare_space_in_memory(self):

        return self.current_memory_size < self.max_memory_size

    def remove_sampled_experiences(self):

        self.memory = self.memory[ :-self.batch_size]
        self.current_memory_size -= self.batch_size

    def __len__(self):
        return len(self.memory)
