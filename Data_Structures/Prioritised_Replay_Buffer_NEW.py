import numpy as np

class Node(object):

    def __init__(self, value, queue_index):
        self.value = value

        self.queue_index = queue_index
        self.heap_index = None

# Queue is most important aspect...



class Prioritised_Replay_Buffer_NEW(object):

    def __init__(self, max_size):

        self.queue = [Node(0, queue_index) for queue_index in range(max_size)] # np.repeat(Node(0), max_size)
        self.queue_index_to_overwrite_next = 0

        self.reached_max_capacity = False


        self.max_size = max_size
        self.heap = [Node(0, None) for _ in range(max_size * 4 + 1)]
        self.heap[0] = Node(float("inf"), None)
        self.heap_index_to_overwrite_next = 1

        self.overall_sum = 0

    def add_element(self, value):

        self.queue[self.queue_index_to_overwrite_next].value = value

        if not self.reached_max_capacity:
            self.heap[self.heap_index_to_overwrite_next] = self.queue[self.queue_index_to_overwrite_next]
            self.queue[self.queue_index_to_overwrite_next].heap_index = self.heap_index_to_overwrite_next
            self.update_heap_index_to_overwrite_next()


        heap_index_change = self.queue[self.queue_index_to_overwrite_next].heap_index
        self.reorganise_heap(heap_index_change)

        self.update_queue_index_to_overwrite_next()

    def update_queue_index_to_overwrite_next(self):

        if self.queue_index_to_overwrite_next < self.max_size - 1:
            self.queue_index_to_overwrite_next += 1
        else:
            self.reached_max_capacity = True
            self.queue_index_to_overwrite_next = 0

    def update_heap_index_to_overwrite_next(self):
        self.heap_index_to_overwrite_next += 1

    def reorganise_heap(self, heap_index_changed):

        node_value = self.heap[heap_index_changed].value
        parent_index = int(heap_index_changed / 2)

        if node_value > self.heap[parent_index].value:
            self.swap(heap_index_changed, parent_index)
            self.reorganise_heap(parent_index)

        else:
            left_child = self.heap[int(heap_index_changed * 2)]
            right_child = self.heap[int(heap_index_changed * 2) + 1]

            if left_child.value > right_child.value:
                biggest_child_index = heap_index_changed * 2
            else:
                biggest_child_index = heap_index_changed * 2 + 1

            if node_value < self.heap[biggest_child_index].value:
                self.swap(heap_index_changed, biggest_child_index)
                self.reorganise_heap(biggest_child_index)

    def swap(self, index1, index2):
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]

        self.heap[index1].heap_index = index1
        self.heap[index2].heap_index = index2

    def give_max_element(self):
        return self.heap[1].value

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