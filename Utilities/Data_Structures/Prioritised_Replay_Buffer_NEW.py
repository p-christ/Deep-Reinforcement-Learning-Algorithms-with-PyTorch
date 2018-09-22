import math
import random

import numpy as np

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



class Prioritised_Replay_Buffer_NEW(object):
    """Data structure that maintains a queue and a heap. The queue keeps track of which experiences are the oldest and so
     tells us which ones to delete once the buffer starts getting full. The heap lets us quickly retrieve the experience
     with the max td_value. We also keep track of the sum of the td values

     Achieves:
     - Extracting max td error - O(1)
     - Extracting sum of td errors - O(1)
     - Updating td errors of sample - O(log N)
     - Add experience - O(log N)
     - Sample experiences - ? 

     """

    def __init__(self, max_buffer_size, batch_size):
        self.max_buffer_size = max_buffer_size

        self.queue = self.initialise_queue()
        self.heap = self.initialise_heap()

        self.queue_index_to_overwrite_next = 0
        self.heap_index_to_overwrite_next = 1
        self.reached_max_capacity = False
        self.overall_sum = 0

        self.batch_size = batch_size

    def initialise_queue(self):
        """Initialises a queue of Nodes of length self.max_size"""
        return [Node(0, None, None, None, None, None, queue_index) for queue_index in range(self.max_buffer_size)]

    def initialise_heap(self):
        """Initialises a heap of Nodes of length self.max_size * 4 + 1"""
        heap = [Node(0, None, None, None, None, None, None) for _ in range(self.max_buffer_size * 4 + 1)]

        # We don't use the 0th element so we want it to have infinite value so it is never swapped with a lower node
        heap[0] = Node(float("inf"), None, None, None, None, None, None)
        return heap

    def add_element(self, td_error, state, action, reward, next_state, done):
        td_error = abs(td_error) + 1e-6
        self.update_overall_sum(td_error)
        self.update_queue(td_error, state, action, reward, next_state, done)
        self.update_heap_and_heap_index_to_overwrite()
        self.update_queue_index_to_overwrite_next()

    def update_overall_sum(self, td_error):
        """Updates the overall sum of td_values present in the buffer"""
        self.overall_sum += td_error - self.queue[self.queue_index_to_overwrite_next].td_error

    def update_queue(self, td_error, state, action, reward, next_state, done):
        """Updates the queue by overwriting the oldest experience with the experience provided"""
        self.queue[self.queue_index_to_overwrite_next].update_experience_values(td_error, state, action, reward,
                                                                                next_state, done)

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

    def give_sample(self):
        """Randomly samples from the experiences in the buffer giving a bias towards experiences higher up the heap
         and therefore experiences more likely to have higher td_errors"""

        sample = []
        heap_indexes_chosen = []
        sample_length = 0
        sum_td_errors = self.give_sum_of_td_errors()

        index = 1

        if self.reached_max_capacity:

            length_buffer = self.max_buffer_size
        else:
            length_buffer = self.queue_index_to_overwrite_next

        while sample_length < self.batch_size:

            prob = self.heap[index].td_error / sum_td_errors

            if random.random() < prob:
                sample.append(self.heap[index])
                heap_indexes_chosen.append(index)
            if index < length_buffer - 1:
                index += 1
            else:
                index = 1

        return sample, heap_indexes_chosen

    def update_td_errors(self, td_errors, heap_indexes):
        for td_error, index in zip(td_errors, heap_indexes):
            self.heap[index].td_error = td_error
            self.reorganise_heap(index)

    def give_max_td_error(self):
        return self.heap[1].td_error

    def give_sum_of_td_errors(self):
        return self.overall_sum

    #
    #
    # def add_element(self, value):
    #
    #
    #
    #     self.update_overall_sum(value)
    #
    #     node = Node(value)
    #
    #     self.heap[self.index_to_overwrite_next] = node
    #     self.reorganise_tree(self.index_to_overwrite_next, node)
    #
    #     self.update_index_to_overwrite_next()
    #
    # def update_overall_sum(self, value):
    #     pass
    #
    # def reorganise_tree(self, index_it_was_added_to, node):
    #

    #

    # def update_index_to_overwrite_next(self):
    #
    #
    # def give_max_value(self):
    #     return self.heap[1].value
    #
    # def give_sum_of_values(self):