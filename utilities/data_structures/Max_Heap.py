import numpy as np
from utilities.data_structures.Node import Node

class Max_Heap(object):
    """Generic max heap object"""
    def __init__(self, max_size, dimension_of_value_attribute, default_key_to_use):

        self.max_size = max_size
        self.dimension_of_value_attribute = dimension_of_value_attribute
        self.default_key_to_use = default_key_to_use
        self.heap = self.initialise_heap()

    def initialise_heap(self):
        """Initialises a heap of Nodes of length self.max_size * 4 + 1"""
        heap = np.array([Node(self.default_key_to_use, tuple([None for _ in range(self.dimension_of_value_attribute)])) for _ in range(self.max_size * 4 + 1)])

        # We don't use the 0th element in a heap so we want it to have infinite value so it is never swapped with a lower node
        heap[0] = Node(float("inf"), (None, None, None, None, None))
        return heap

    def update_element_and_reorganise_heap(self, heap_index_for_change, new_element):
        self.update_heap_element(heap_index_for_change, new_element)
        self.reorganise_heap(heap_index_for_change)

    def update_heap_element(self, heap_index, new_element):
        self.heap[heap_index] = new_element

    def reorganise_heap(self, heap_index_changed):
        """This reorganises the heap after a new value is added so as to keep the max value at the top of the heap which
        is index position 1 in the array self.heap"""

        node_key = self.heap[heap_index_changed].key
        parent_index = int(heap_index_changed / 2)

        if node_key > self.heap[parent_index].key:
            self.swap_heap_elements(heap_index_changed, parent_index)
            self.reorganise_heap(parent_index)

        else:
            biggest_child_index = self.calculate_index_of_biggest_child(heap_index_changed)
            if node_key < self.heap[biggest_child_index].key:
                self.swap_heap_elements(heap_index_changed, biggest_child_index)
                self.reorganise_heap(biggest_child_index)

    def swap_heap_elements(self, index1, index2):
        """Swaps the position of two heap elements"""
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]

    def calculate_index_of_biggest_child(self, heap_index_changed):
        """Calculates the heap index of the node's child with the biggest td_error value"""
        left_child = self.heap[int(heap_index_changed * 2)]
        right_child = self.heap[int(heap_index_changed * 2) + 1]

        if left_child.key > right_child.key:
            biggest_child_index = heap_index_changed * 2
        else:
            biggest_child_index = heap_index_changed * 2 + 1

        return biggest_child_index

    def give_max_key(self):
        """Returns the maximum td error currently in the heap. Because it is a max heap this is the top element of the heap"""
        return self.heap[1].key
