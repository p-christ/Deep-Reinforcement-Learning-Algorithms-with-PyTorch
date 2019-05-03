import numpy as np
from utilities.data_structures.Node import Node

class Deque(object):
    """Generic deque object"""
    def __init__(self, max_size, dimension_of_value_attribute):

        self.max_size = max_size
        self.dimension_of_value_attribute = dimension_of_value_attribute
        self.deque = self.initialise_deque()
        self.deque_index_to_overwrite_next = 0
        self.reached_max_capacity = False
        self.number_experiences_in_deque = 0

    def initialise_deque(self):
        """Initialises a queue of Nodes of length self.max_size"""
        deque = np.array([Node(0, tuple([None for _ in range(self.dimension_of_value_attribute)])) for _ in range(self.max_size)])
        return deque

    def add_element_to_deque(self, new_key, new_value):
        """Adds an element to the deque and then updates the index of the next element to be overwritten and also the
        amount of elements in the deque"""
        self.update_deque_node_key_and_value(self.deque_index_to_overwrite_next, new_key, new_value)
        self.update_number_experiences_in_deque()
        self.update_deque_index_to_overwrite_next()

    def update_deque_node_key_and_value(self, index, new_key, new_value):
        self.update_deque_node_key(index, new_key)
        self.update_deque_node_value(index, new_value)

    def update_deque_node_key(self, index, new_key):
        self.deque[index].update_key(new_key)

    def update_deque_node_value(self, index, new_value):
        self.deque[index].update_value(new_value)

    def update_deque_index_to_overwrite_next(self):
        """Updates the deque index that we should write over next. When the buffer gets full we begin writing over
         older experiences"""
        if self.deque_index_to_overwrite_next < self.max_size - 1:
            self.deque_index_to_overwrite_next += 1
        else:
            self.reached_max_capacity = True
            self.deque_index_to_overwrite_next = 0

    def update_number_experiences_in_deque(self):
        """Keeps track of how many experiences there are in the buffer"""
        if not self.reached_max_capacity:
            self.number_experiences_in_deque += 1