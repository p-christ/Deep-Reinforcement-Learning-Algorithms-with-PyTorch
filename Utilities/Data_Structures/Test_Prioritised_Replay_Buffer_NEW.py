import random
import numpy as np
from Prioritised_Replay_Buffer_NEW import Prioritised_Replay_Buffer_NEW


def test_prioritised_replay_buffer_add_in_decreasing_elements():

    buffer = Prioritised_Replay_Buffer_NEW(max_buffer_size=3, batch_size=None)

    buffer.add_element(100, None, None, None, None, None)

    assert buffer.queue[0].td_error == 100
    assert buffer.queue[0].heap_index == 1
    assert buffer.heap[1].td_error == 100

    buffer.add_element(99, None, None, None, None, None)
    buffer.add_element(98, None, None, None, None, None)

    assert buffer.queue[1].td_error == 99
    assert buffer.queue[1].heap_index == 2
    assert buffer.heap[2].td_error == 99
    assert buffer.queue[2].td_error == 98
    assert buffer.queue[2].heap_index == 3
    assert buffer.heap[3].td_error == 98

    buffer.add_element(97, None, None, None, None, None)

    assert buffer.queue[0].td_error == 97
    assert buffer.queue[0].heap_index == 2
    assert buffer.heap[2].td_error == 97
    assert buffer.heap[1].td_error == 99

    buffer.add_element(96, None, None, None, None, None)

    assert buffer.queue[1].td_error == 96
    assert buffer.queue[1].heap_index == 3
    assert buffer.heap[3].td_error == 96


def test_prioritised_replay_buffer_add_in_increasing_elements():

    buffer = Prioritised_Replay_Buffer_NEW(max_buffer_size=3, batch_size=None)

    buffer.add_element(100, None, None, None, None, None)

    assert buffer.queue[0].td_error == 100
    assert buffer.queue[0].heap_index == 1
    assert buffer.heap[1].td_error == 100

    buffer.add_element(101, None, None, None, None, None)
    buffer.add_element(102, None, None, None, None, None)

    assert buffer.queue[0].td_error == 100
    assert buffer.queue[1].td_error == 101
    assert buffer.queue[2].td_error == 102

    assert buffer.heap[1].td_error == 102
    assert buffer.heap[2].td_error == 100
    assert buffer.heap[3].td_error == 101

    assert buffer.queue[0].heap_index == 2
    assert buffer.queue[1].heap_index == 3
    assert buffer.queue[2].heap_index == 1

    buffer.add_element(103, None, None, None, None, None)

    assert buffer.queue[0].td_error == 103

    assert buffer.heap[1].td_error == 103
    assert buffer.heap[2].td_error == 102

    assert buffer.queue[0].heap_index == 1
    assert buffer.queue[1].heap_index == 3
    assert buffer.queue[2].heap_index == 2

    buffer.add_element(104, None, None, None, None, None)

    assert buffer.queue[1].td_error == 104
    assert buffer.heap[1].td_error == 104
    assert buffer.heap[3].td_error == 103

    assert buffer.queue[0].heap_index == 3
    assert buffer.queue[1].heap_index == 1
    assert buffer.queue[2].heap_index == 2


def test_give_max_element_always_keeps_max_at_top():
    max_buffer_size = 50
    for _ in range(1000):
        buffer = Prioritised_Replay_Buffer_NEW(max_buffer_size=max_buffer_size, batch_size=None)
        elements_added = []
        for _ in range(100):
            element = round(random.random(), 6)
            elements_added.append(element)
            buffer.add_element(element, None, None, None, None, None)

        max_td_error = np.max(elements_added[-max_buffer_size:])
        assert buffer.give_max_td_error() == max_td_error, "{}".format(elements_added)

def test_give_sum_of_elements_is_always_correct():
    max_buffer_size = 50
    round_to_this_many_decimal_places = 6
    for _ in range(1000):
        buffer = Prioritised_Replay_Buffer_NEW(max_buffer_size=max_buffer_size, batch_size=None)
        elements_added = []
        for _ in range(100):
            element = round(random.random())
            elements_added.append(element)
            buffer.add_element(element, None, None, None, None, None)

        sum_td_error = np.sum(elements_added[-max_buffer_size:])
        assert round(buffer.give_sum_of_td_errors(), round_to_this_many_decimal_places) == round(sum_td_error, round_to_this_many_decimal_places), "{}".format(elements_added)

def test_give_num_layers_in_heap():

    buffer = Prioritised_Replay_Buffer_NEW(max_buffer_size=10, batch_size=None)

    buffer.add_element(1, None, None, None, None, None)

    assert buffer.give_num_layers_in_heap() == 1

    buffer.add_element(2, None, None, None, None, None)
    assert buffer.give_num_layers_in_heap() == 2

    buffer.add_element(3, None, None, None, None, None)
    assert buffer.give_num_layers_in_heap() == 2

    buffer.add_element(4, None, None, None, None, None)
    assert buffer.give_num_layers_in_heap() == 3
    buffer.add_element(5, None, None, None, None, None)
    assert buffer.give_num_layers_in_heap() == 3

    buffer.add_element(6, None, None, None, None, None)
    buffer.add_element(7, None, None, None, None, None)
    assert buffer.give_num_layers_in_heap() == 3

    buffer.add_element(8, None, None, None, None, None)
    assert buffer.give_num_layers_in_heap() == 4


def test_give_sample_indexes():
    batch_size = 3
    buffer = Prioritised_Replay_Buffer_NEW(max_buffer_size=5, batch_size=batch_size)

    buffer.add_element(1, None, None, None, None, None)
    buffer.add_element(1, None, None, None, None, None)
    buffer.add_element(1, None, None, None, None, None)
    buffer.add_element(1, None, None, None, None, None)
    buffer.add_element(1, None, None, None, None, None)

    indexes = buffer.give_sample_indexes()

    assert 1 in indexes
    assert 2 in indexes or 3 in indexes
    assert not (2 in indexes and 3 in indexes)
    assert 4 in indexes or 5 in indexes
    assert not (4 in indexes and 5 in indexes)
    assert len(indexes) == batch_size

