from collections import namedtuple, deque
import random
import torch
import numpy as np
from segment_tree import *
import time



""" Note they don't calculate the td error for each new observation straight away, they instead put it to front of queue for sampling """



#
class Prioritised_Replay_Buffer_Segment_Tree_Impl(object):

    def __init__(self, max_buffer_size, batch_size, seed, alpha, beta, state_size):
        self.max_buffer_size = max_buffer_size
        # self.memory = deque(maxlen=max_buffer_size)
        # self.memory = np.zeros((self.max_buffer_size, 6))

        self.state_memory = np.zeros((self.max_buffer_size, state_size))
        self.action_memory = np.zeros((self.max_buffer_size, 1))
        self.reward_memory = np.zeros((self.max_buffer_size, 1))
        self.next_state_memory = np.zeros((self.max_buffer_size, state_size))
        self.done_memory = np.zeros((self.max_buffer_size, 1))
        self.td_error_memory = np.zeros((self.max_buffer_size, 1))

        self.batch_size = batch_size
        # self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "td_error"])

        # self.experience = np.zeros((self.max_buffer_size, 6))
        self.seed = random.seed(seed)

        self.segment_tree = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.alpha = alpha
        self.beta = beta
        self.next_index_to_overwrite = 0

        self.max_importance_sampling_value = 1

        self.segment_tree = SegmentTree(np.zeros(self.max_buffer_size))

        self.current_buffer_size = 0



    def add(self, state, action, reward, next_state, done):

        if self.current_buffer_size == 0:
            adapted_td_error = 1

        else:
            adapted_td_error = self.segment_tree.query(0, self.current_buffer_size - 1, "max") #set td_error for new entry to max value

        self.state_memory[self.next_index_to_overwrite] = state
        self.action_memory[self.next_index_to_overwrite] = action
        self.reward_memory[self.next_index_to_overwrite] = reward
        self.next_state_memory[self.next_index_to_overwrite] = next_state
        self.done_memory[self.next_index_to_overwrite] = done
        self.td_error_memory[self.next_index_to_overwrite] = adapted_td_error

        self.current_buffer_size = min(self.current_buffer_size + 1, self.max_buffer_size)

        self.segment_tree.update(self.next_index_to_overwrite, adapted_td_error)
        self.update_next_index_to_overwrite()



    def memory_reached_max_buffer_size(self):
        return self.current_buffer_size >= self.max_buffer_size

    def update_next_index_to_overwrite(self):

        if self.next_index_to_overwrite < self.max_buffer_size - 1:
            self.next_index_to_overwrite += 1
        else:
            self.next_index_to_overwrite = 0



    def sample(self):
        sample_indexes, sample_probabilities, sample_experiences = self.pick_experiences()
        states, actions, rewards, next_states, dones = self.separate_out_data_types(sample_experiences)

        importance_sampling_weights = self.calculate_importance_sampling_weights(sample_probabilities, sample_indexes)

        # return (sample_indexes, importance_sampling_weights, (states, actions, rewards, next_states, dones))

        return (sample_indexes, importance_sampling_weights, (states, actions, rewards, next_states, dones))

    def calculate_importance_sampling_weights(self, sample_probabilities, sample_indexes):


        sum_of_td_errors = np.sum(self.td_error_memory)
        max_td_error = np.max(self.td_error_memory)
        #
        # sum_of_td_errors = self.segment_tree.query(0, self.current_buffer_size - 1, "sum")
        # max_td_error = self.segment_tree.query(0, self.current_buffer_size - 1, "max")
        max_probability = max_td_error / sum_of_td_errors

        self.max_importance_sampling_value = ((max_probability * self.current_buffer_size) ** -self.beta) / self.max_importance_sampling_value

        adapted_weights = sample_probabilities * self.current_buffer_size
        adapted_weights = np.power(adapted_weights, -self.beta)

        importance_sampling_weights = adapted_weights  / self.max_importance_sampling_value

        importance_sampling_weights = importance_sampling_weights.reshape(importance_sampling_weights.shape[0], 1)

        print(sample_probabilities.shape)
        print(importance_sampling_weights.shape)
        print(sample_indexes.shape)

        print(sample_indexes)

        print(self.current_buffer_size)

        importance_sampling_weights[sample_indexes]

        importance_sampling_weights = importance_sampling_weights[sample_indexes]



        return importance_sampling_weights



    def separate_out_data_types(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        # ["state", "action", "reward", "next_state", "done", "td_error"])
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)



        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
        #     self.device)
        # dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def pick_experiences(self):

        start = time.time()
        # sum_of_td_errors = self.segment_tree.query(0, self.current_buffer_size - 1, "sum")

        sum_of_td_errors = np.sum(self.td_error_memory)


        probabilities = self.td_error_memory[:self.current_buffer_size, 0] / sum_of_td_errors


        sample_indexes = np.random.choice(self.current_buffer_size, size=self.batch_size, p=probabilities, replace=False)

        sample_states = self.state_memory[sample_indexes]
        sample_actions = self.action_memory[sample_indexes]
        sample_rewards = self.reward_memory[sample_indexes]
        sample_next_states = self.next_state_memory[sample_indexes]
        sample_dones = self.done_memory[sample_indexes]

        sample_probabilities = probabilities[sample_indexes]


        # if self.current_buffer_size % 100 == 0:
        #
        #     print("   Pick experience takes: ", time.time() - start)
        #


        return sample_indexes, sample_probabilities, (sample_states, sample_actions, sample_rewards, sample_next_states, sample_dones)


    def update_td_errors(self, indexes_to_update, td_errors):

        start = time.time()

        adapted_td_errors = abs(td_errors.reshape((td_errors.shape[0], 1))) + 1e-8 ** self.alpha

        self.td_error_memory[indexes_to_update] = adapted_td_errors

        for td_errors_index, index_to_update in enumerate(indexes_to_update):
            self.segment_tree.update(index_to_update, adapted_td_errors[td_errors_index])

        # td_error_values =  abs(td_errors) + 1e-8 ** self.alpha

        # for td_errors_index, index_to_update in enumerate(indexes_to_update):
        #
        #     td_error = (abs(td_errors[td_errors_index]) + 1e-8) ** self.alpha #add a small amount so no td_errors equal 0
        #
        #     self.segment_tree.update(index_to_update, td_error)
        #     # experience = self.memory[index_to_update]
        #     self.td_error_memory[index_to_update, 0] = td_error
        #
        #     # self.memory[index_to_update] = self.experience(experience.state, experience.action, experience.reward,
        #     #                                             experience.next_state, experience.done,
        #     #                                                td_error)
        #     #

        # if self.current_buffer_size % 100 == 0:
        #
        #     print("   Update td errors takes: ", time.time() - start)


    def __len__(self):
        return self.current_buffer_size


    #need an update td_error method\

