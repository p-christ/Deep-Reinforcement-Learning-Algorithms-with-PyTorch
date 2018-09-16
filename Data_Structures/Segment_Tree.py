
import numpy as np

# better to use an array than a node class
# with an array do it so:   2k is left child and 2k+1 is right child for node k
# should make root node 1 rather than 0 so that the above 2k 2k+1 rule works (it won't work with 0)

# need keep track of:

# All td errors
# so that can easily extract:
# 1) sum of td errors
# 2) the smallest td error
#
# and can easily update td_Errors
#
#
# 1) sum of td_errors
# 2) the smallest td error
# 3) all td errors
#
# in a way


# Slow version of data structure to work as test:


class Slow_Segment_Tree(object):

    def __init__(self):
        self.values = []

    def increment(self, start_index, end_index_inclusive, increment_value):
        for index in (start_index, end_index_inclusive + 1):
            self.values[index] += increment_value

    def get_minimum_within_range(self, start_index, end_index_inclusive):

        min_value = float("inf")
        for index in (start_index, end_index_inclusive + 1):
            min_value = min(min_value, self.values[index])

        return min_value


class Min_Segment_Tree(object):

    def __init__(self, capacity, neutral_element=float("inf")):

        self.capacity = capacity
        self.values = [neutral_element for _ in range(2 * self.capacity)]

        self.low_ranges_index_responsible_for = np.zeros(4 * self.capacity + 1) # index i in this array tells you the lower inclusive index for nodes that index i represents
        self.high_ranges_index_responsible_for = np.zeros(4 * self.capacity + 1) # index i in this array tells you the higher inclusive index for nodes that index i represents

        self.set_range_of_indexes_responsible_for(1, 0, self.capacity - 1)

        self.min_over_subtree_range = [neutral_element for _ in range(4 * self.capacity + 1)]
        self.delta_that_needs_to_be_pushed_down = np.zeros(4 * self.capacity + 1)  # this stores the changes that needs to be pushed down once the node gets updated

    def set_range_of_indexes_responsible_for(self, node_index, start_index, end_index_inclusive):
        self.low_ranges_index_responsible_for[node_index] = start_index
        self.high_ranges_index_responsible_for[node_index] = end_index_inclusive

        if start_index == end_index_inclusive:  #we are at a leaf if this is true
            return

        split_point_index = int((end_index_inclusive - start_index) / 2)
        self.set_range_of_indexes_responsible_for(2*node_index, start_index, split_point_index) # set indexes for left child
        self.set_range_of_indexes_responsible_for(2*node_index + 1, split_point_index + 1, end_index_inclusive) # set indexes for right child

    def increment(self, node_index, start_index_to_increment, end_index_inclusive_to_increment, increment_value):

        if self.increment_range_outside_range_covered_by_this_node(node_index, start_index_to_increment, end_index_inclusive_to_increment):
            return

        if self.increment_range_completely_covers_range_covered_by_this_node(node_index, start_index_to_increment, end_index_inclusive_to_increment):
            self.delta_that_needs_to_be_pushed_down[node_index] += increment_value
            return

        self.propagate_delta_downwards(node_index)

        self.increment(node_index * 2, start_index_to_increment, end_index_inclusive_to_increment, increment_value)
        self.increment(node_index * 2 + 1, start_index_to_increment, end_index_inclusive_to_increment, increment_value)

        self.update_range_minimum(node_index)


        # will be changes to pass down and to pass up


    def increment_range_outside_range_covered_by_this_node(self, node_index, start_index_to_increment, end_index_inclusive_to_increment):
        return end_index_inclusive_to_increment < self.low_ranges_index_responsible_for[node_index] or start_index_to_increment > self.high_ranges_index_responsible_for[node_index]

    def increment_range_completely_covers_range_covered_by_this_node(self, node_index, start_index_to_increment, end_index_inclusive_to_increment):
        return start_index_to_increment <= self.low_ranges_index_responsible_for[node_index] and end_index_inclusive_to_increment >= self.high_ranges_index_responsible_for[node_index]

    #
    # def increment_range_completely_inside_range_covered_by_this_node(self, node_index, start_index_to_increment, end_index_inclusive_to_increment):
    #     end_index_inclusive_to_increment <= self.high_ranges_index_responsible_for[node_index] and start_index_to_increment >= self.low_ranges_index_responsible_for[node_index]



    def propagate_delta_downwards(self, node_index):
        self.delta_that_needs_to_be_pushed_down[2 * node_index] += self.delta_that_needs_to_be_pushed_down[node_index]
        self.delta_that_needs_to_be_pushed_down[2 * node_index + 1] += self.delta_that_needs_to_be_pushed_down[node_index]
        self.delta_that_needs_to_be_pushed_down[node_index] = 0

    def update_range_minimum(self, node_index):

        self.min_over_subtree_range[node_index] = min(self.min_over_subtree_range[2 * node_index] + self.delta_that_needs_to_be_pushed_down[2 * node_index],
                                                      self.min_over_subtree_range[2 * node_index + 1] + self.delta_that_needs_to_be_pushed_down[2 * node_index + 1])



# initi function = for a node id, what is the range of nodes it is responsible for?
# lo = low end of range responsible for, hi = high end of range responsible for

