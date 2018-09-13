from collections import namedtuple, deque
import random
import torch
import numpy as np
from segment_tree import *




""" Note they don't calculate the td error for each new observation straight away, they instead put it to front of queue for sampling """



#
class Prioritised_Replay_Buffer_Segment_Tree_Impl(object):

    def __init__(self, max_buffer_size, batch_size, seed, alpha, beta):
        self.max_buffer_size = max_buffer_size
        # self.memory = deque(maxlen=max_buffer_size)
        self.memory = []
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "td_error"])
        self.seed = random.seed(seed)

        self.segment_tree = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.alpha = alpha
        self.beta = beta
        self.next_index_to_overwrite = 0

        self.max_importance_sampling_value = 1

        self.segment_tree = SegmentTree(np.zeros(self.max_buffer_size))



    def add(self, state, action, reward, next_state, done):

        if len(self.memory) == 0:
            adapted_td_error = 1

        else:
            adapted_td_error = self.segment_tree.query(0, len(self.memory) - 1, "max") #set td_error for new entry to max value

        experience = self.experience(state, action, reward, next_state, done, adapted_td_error)

        if self.memory_reached_max_buffer_size():
            self.memory[self.next_index_to_overwrite] = experience
        else:
            self.memory.append(experience)

        self.segment_tree.update(self.next_index_to_overwrite, adapted_td_error)
        self.update_next_index_to_overwrite()

    def memory_reached_max_buffer_size(self):
        return len(self.memory) >= self.max_buffer_size

    def update_next_index_to_overwrite(self):

        if self.next_index_to_overwrite < self.max_buffer_size - 1:
            self.next_index_to_overwrite += 1
        else:
            self.next_index_to_overwrite = 0



    def sample(self):
        sample_indexes, sample_probabilities, sample_experiences = self.pick_experiences()
        states, actions, rewards, next_states, dones = self.separate_out_data_types(sample_experiences)

        importance_sampling_weights = self.calculate_importance_sampling_weights(sample_probabilities)

        return (sample_indexes, importance_sampling_weights, (states, actions, rewards, next_states, dones))

    def calculate_importance_sampling_weights(self, sample_probabilities):

        sum_of_td_errors = self.segment_tree.query(0, len(self.memory) - 1, "sum")
        max_td_error = self.segment_tree.query(0, len(self.memory) - 1, "max")
        max_probability = max_td_error / sum_of_td_errors

        self.max_importance_sampling_value = ((max_probability * len(self.memory)) ** -self.beta) / self.max_importance_sampling_value

        adapted_weights = sample_probabilities * len(self.memory)
        adapted_weights = [adapted_weight ** -self.beta for adapted_weight in adapted_weights]

        importance_sampling_weights = adapted_weights  / self.max_importance_sampling_value

        return importance_sampling_weights



    def separate_out_data_types(self, experiences):
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def pick_experiences(self):

        sum_of_td_errors = self.segment_tree.query(0, len(self.memory) - 1, "sum")
        probabilities = [(experience.td_error) / sum_of_td_errors for experience in self.memory]

        sample_indexes = np.random.choice(len(self.memory), size=self.batch_size, p=probabilities, replace=False)

        sample_experiences = []
        sample_probabilities = []

        for index in sample_indexes:
            sample_experiences.append(self.memory[index])
            sample_probabilities.append(probabilities[index])

        return sample_indexes, sample_probabilities, sample_experiences
        #
        # return random.sample(self.memory, k=self.batch_size)

    # def pick_indexes(self):
    #     possible_indexes = len(self.memory)
    #
    #     sum_of_td_errors = self.segment_tree.query(0, len(self.memory) - 1, "sum")
    #
    #     sample_probabilities = [experience.td_error ** self.alpha / sum_of_td_errors for experience in self.memory]
    #
    #     chosen_indexes = np.random.choice(len(self.memory), size=self.batch_size, p=sample_probabilities, replace=False)
    #     return chosen_indexes

    def update_td_errors(self, indexes_to_update, td_errors):

        for td_errors_index, index_to_update in enumerate(indexes_to_update):

            td_error = (abs(td_errors[td_errors_index]) + 1e-8) ** self.alpha #add a small amount so no td_errors equal 0

            self.segment_tree.update(index_to_update, td_error)
            experience = self.memory[index_to_update]
            self.memory[index_to_update] = self.experience(experience.state, experience.action, experience.reward,
                                                        experience.next_state, experience.done,
                                                           td_error)



    def __len__(self):
        return len(self.memory)


    #need an update td_error method