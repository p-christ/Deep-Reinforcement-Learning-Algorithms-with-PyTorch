import math
import random
from operator import itemgetter
import numpy as np
import torch

from Utilities.Data_Structures.Replay_Buffer import Replay_Buffer


class Prioritised_Replay_Buffer(Replay_Buffer):
    """Data structure that maintains a queue and a heap. The queue keeps track of which experiences are the oldest and so
     tells us which ones to delete once the buffer starts getting full. The heap lets us quickly retrieve the experience
     with the max td_value. We also keep track of the sum of the td values

     Complexity:
     - Extracting max td error - O(1)
     - Extracting sum of td errors - O(1)
     - Updating td errors of sample - O(log N)
     - Add experience - O(log N)
     - Sample experiences - O(1)

     """

    def __init__(self, hyperparameters, seed=0):
        hyperparameters = hyperparameters["DQN_Agents"]
        Replay_Buffer.__init__(self, hyperparameters["buffer_size"], hyperparameters["batch_size"], seed=0)
        np.random.seed(seed)
        self.max_buffer_size = hyperparameters["buffer_size"]



        self.queue = self.initialise_queue()
        self.queues_td_errors = self.initialise_td_errors_array()
        self.heap = self.initialise_heap()

        self.queue_index_to_overwrite_next = 0
        self.heap_index_to_overwrite_next = 1
        self.number_experiences_in_buffer = 0
        self.reached_max_capacity = False
        self.adapted_overall_sum_of_td_errors = 0

        self.alpha = hyperparameters["alpha_prioritised_replay"]
        self.beta = hyperparameters["beta_prioritised_replay"]
        self.incremental_td_error = hyperparameters["incremental_td_error"]


        self.batch_size = hyperparameters["batch_size"]

        self.heap_indexes_to_update_td_error_for = None

    def initialise_queue(self):
        """Initialises a queue of Nodes of length self.max_size"""
        return np.array([Node(0, None, None, None, None, None, queue_index) for queue_index in range(self.max_buffer_size)])

    def initialise_td_errors_array(self):
        """Initialises a queue of Nodes of length self.max_size"""
        return np.zeros(self.max_buffer_size)


    def initialise_heap(self):
        """Initialises a heap of Nodes of length self.max_size * 4 + 1"""
        heap = np.array([Node(0, None, None, None, None, None, None) for _ in range(self.max_buffer_size * 4 + 1)])

        # We don't use the 0th element in a heap so we want it to have infinite value so it is never swapped with a lower node
        heap[0] = Node(float("inf"), None, None, None, None, None, None)
        return heap

    def add_experience(self, raw_td_error, state, action, reward, next_state, done):
        td_error = (abs(raw_td_error) + self.incremental_td_error) ** self.alpha
        self.update_overall_sum(td_error, self.queue[self.queue_index_to_overwrite_next].td_error)
        self.update_queue_and_queue_td_errors(td_error, state, action, reward, next_state, done)
        self.update_number_experiences_in_buffer()
        self.update_heap_and_heap_index_to_overwrite()
        self.update_queue_index_to_overwrite_next()

    def update_overall_sum(self, new_td_error, old_td_error):
        """Updates the overall sum of td_values present in the buffer"""
        self.adapted_overall_sum_of_td_errors += new_td_error  - old_td_error

    def update_queue_and_queue_td_errors(self, td_error, state, action, reward, next_state, done):
        """Updates the queue by overwriting the oldest experience with the experience provided"""
        self.queue[self.queue_index_to_overwrite_next].update_experience_values(td_error, state, action, reward,
                                                                                next_state, done)
        self.queues_td_errors[self.queue_index_to_overwrite_next] = td_error

    def update_heap_and_heap_index_to_overwrite(self):
        """Updates the heap by rearranging it given the new experience that was just incorporated into it. If we haven't
        reached max capacity then the new experience is added directly into the heap, otherwise a pointer on the heap has
        changed to reflect the new experience so there's no need to add it in"""
        if not self.reached_max_capacity:
            self.heap[self.heap_index_to_overwrite_next] = self.queue[self.queue_index_to_overwrite_next]
            self.queue[self.queue_index_to_overwrite_next].heap_index = self.heap_index_to_overwrite_next
            self.update_heap_index_to_overwrite_next()

        heap_index_change = self.queue[self.queue_index_to_overwrite_next].heap_index
        self.reorganise_heap(heap_index_change)


    def update_queue_index_to_overwrite_next(self):
        """Updates the queue index that we should write over next. When the buffer gets full we begin writing over
         older experiences"""
        if self.queue_index_to_overwrite_next < self.max_buffer_size - 1:
            self.queue_index_to_overwrite_next += 1
        else:
            self.reached_max_capacity = True
            self.queue_index_to_overwrite_next = 0

    def update_heap_index_to_overwrite_next(self):
        """This updates the heap index to write over next. Once the buffer gets full we stop calling this function because
        the nodes the heap points to start being changed directly rather than the pointers on the heap changing"""
        self.heap_index_to_overwrite_next += 1

    def update_number_experiences_in_buffer(self):
        """Keeps track of how many experiences there are in the buffer"""
        if not self.reached_max_capacity:
            self.number_experiences_in_buffer += 1

    def reorganise_heap(self, heap_index_changed):
        """This reorganises the heap after a new value is added so as to keep the max value at the top of the heap which
        is index position 1 in the array self.heap"""
        node_td_error = self.heap[heap_index_changed].td_error
        parent_index = int(heap_index_changed / 2)

        if node_td_error > self.heap[parent_index].td_error:
            self.swap_heap_elements_and_update_node_heap_indexes(heap_index_changed, parent_index)
            self.reorganise_heap(parent_index)

        else:
            left_child = self.heap[int(heap_index_changed * 2)]
            right_child = self.heap[int(heap_index_changed * 2) + 1]

            if left_child.td_error > right_child.td_error:
                biggest_child_index = heap_index_changed * 2
            else:
                biggest_child_index = heap_index_changed * 2 + 1

            if node_td_error < self.heap[biggest_child_index].td_error:
                self.swap_heap_elements_and_update_node_heap_indexes(heap_index_changed, biggest_child_index)
                self.reorganise_heap(biggest_child_index)

    def swap_heap_elements_and_update_node_heap_indexes(self, index1, index2):
        """Swaps two heap elements position and then updates the heap_index stored in the two nodes"""
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]
        self.heap[index1].heap_index = index1
        self.heap[index2].heap_index = index2

    def sample(self, rank_based=True):
        """Randomly samples from the experiences in the buffer giving a bias towards experiences higher up the heap
         and therefore experiences more likely to have higher td_errors. Specifically it randomly picks 1 experience from each level
         of the tree (starting at the top) until it has created a sample of big enough batch size"""
        experiences, queue_sample_indexes = self.pick_experiences_based_on_proportional_td_error()
        states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
        self.queue_sample_indexes_to_update_td_error_for = queue_sample_indexes

        td_errors = [experience.td_error for experience in experiences]

        importance_sampling_weights = [((1.0 / self.number_experiences_in_buffer) * (self.give_adapted_sum_of_td_errors() / td_error)) ** self.beta for td_error in td_errors]

        max_is_weight = max(importance_sampling_weights)

        importance_sampling_weights = [is_weight / max_is_weight for is_weight in importance_sampling_weights]

        importance_sampling_weights = torch.tensor(importance_sampling_weights).float().to(self.device)

        return (states, actions, rewards, next_states, dones), importance_sampling_weights


    def pick_experiences_based_on_proportional_td_error(self):

        probabilities = self.queues_td_errors / self.give_adapted_sum_of_td_errors()
        queue_sample_indexes = np.random.choice(range(len(self.queues_td_errors)), size=self.batch_size, replace=False, p=probabilities)
        experiences = self.queue[queue_sample_indexes]

        return experiences, queue_sample_indexes

    def update_td_errors(self, td_errors):
        """Updates the td_errors for the provided heap indexes. The indexes should be the observations provided most
        recently by the give_sample method"""
        for raw_td_error, queue_index in zip(td_errors, self.queue_sample_indexes_to_update_td_error_for):
            td_error =  (abs(raw_td_error) + self.incremental_td_error) ** self.alpha
            corresponding_heap_index = self.queue[queue_index].heap_index
            self.update_overall_sum(td_error, self.heap[corresponding_heap_index].td_error)
            self.heap[corresponding_heap_index].td_error = td_error
            self.reorganise_heap(corresponding_heap_index)
            self.queues_td_errors[queue_index] = td_error

    def give_max_td_error(self):
        """Returns the maximum td error currently in the heap. Because it is a max heap this is the top element of the heap"""
        return self.heap[1].td_error

    def give_adapted_sum_of_td_errors(self):
        """Returns the sum of td errors of the experiences currently in the heap"""
        return self.adapted_overall_sum_of_td_errors

    def __len__(self):
        return self.number_experiences_in_buffer


class Node(object):

    def __init__(self, td_error, state, action, reward, next_state, done, queue_index):
        self.td_error = td_error
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

        self.queue_index = queue_index
        self.heap_index = None

    def update_experience_values(self, td_error, state, action, reward, next_state, done):
        self.td_error = td_error
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


