import random
from utilities.data_structures.Max_Heap import Max_Heap
import numpy as np
from utilities.data_structures.Node import Node


def test_heap_always_keeps_max_element_at_top():
    max_heap_size = 200
    for _ in range(100):
        heap = Max_Heap(max_heap_size, 2, 0)
        elements_added = []
        for ix in range(1, 100):
            element = random.random()
            elements_added.append(element)
            heap.update_element_and_reorganise_heap(ix, Node(element, (None, None)))

        max_key = np.max(elements_added)
        assert round(heap.give_max_key(), 8) == round(max_key, 8), "{}".format(elements_added)

