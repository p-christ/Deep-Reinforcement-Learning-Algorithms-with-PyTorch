import random
import numpy as np
from Prioritised_Replay_Buffer_NEW import Prioritised_Replay_Buffer_NEW


def test_prioritised_replay_buffer_add_in_decreasing_elements():

    buffer = Prioritised_Replay_Buffer_NEW(max_size=3)

    buffer.add_element(100)

    assert buffer.queue[0].value == 100
    assert buffer.queue[0].heap_index == 1
    assert buffer.heap[1].value == 100

    buffer.add_element(99)
    buffer.add_element(98)

    assert buffer.queue[1].value == 99
    assert buffer.queue[1].heap_index == 2
    assert buffer.heap[2].value == 99
    assert buffer.queue[2].value == 98
    assert buffer.queue[2].heap_index == 3
    assert buffer.heap[3].value == 98

    buffer.add_element(97)

    assert buffer.queue[0].value == 97
    assert buffer.queue[0].heap_index == 2
    assert buffer.heap[2].value == 97
    assert buffer.heap[1].value == 99

    buffer.add_element(96)

    assert buffer.queue[1].value == 96
    assert buffer.queue[1].heap_index == 3
    assert buffer.heap[3].value == 96


def test_prioritised_replay_buffer_add_in_increasing_elements():

    buffer = Prioritised_Replay_Buffer_NEW(max_size=3)

    buffer.add_element(100)

    assert buffer.queue[0].value == 100
    assert buffer.queue[0].heap_index == 1
    assert buffer.heap[1].value == 100

    buffer.add_element(101)
    buffer.add_element(102)

    assert buffer.queue[0].value == 100
    assert buffer.queue[1].value == 101
    assert buffer.queue[2].value == 102

    assert buffer.heap[1].value == 102
    assert buffer.heap[2].value == 100
    assert buffer.heap[3].value == 101

    assert buffer.queue[0].heap_index == 2
    assert buffer.queue[1].heap_index == 3
    assert buffer.queue[2].heap_index == 1

    buffer.add_element(103)

    assert buffer.queue[0].value == 103

    assert buffer.heap[1].value == 103
    assert buffer.heap[2].value == 102

    assert buffer.queue[0].heap_index == 1
    assert buffer.queue[1].heap_index == 3
    assert buffer.queue[2].heap_index == 2

    buffer.add_element(104)

    assert buffer.queue[1].value == 104
    assert buffer.heap[1].value == 104
    assert buffer.heap[3].value == 103

    assert buffer.queue[0].heap_index == 3
    assert buffer.queue[1].heap_index == 1
    assert buffer.queue[2].heap_index == 2


def test_give_max_element_always_keeps_max_at_top():
    max_buffer_size = 50
    for _ in range(1000):
        buffer = Prioritised_Replay_Buffer_NEW(max_size=max_buffer_size)
        elements_added = []
        for _ in range(100):
            element = round(random.random(), 6)
            elements_added.append(element)
            buffer.add_element(element)

        max_value = np.max(elements_added[-max_buffer_size:])
        assert buffer.give_max_element() == max_value, "{}".format(elements_added)

def test_give_sum_of_elements_is_always_correct():
    max_buffer_size = 50
    round_to_this_many_decimal_places = 6
    for _ in range(1000):
        buffer = Prioritised_Replay_Buffer_NEW(max_size=max_buffer_size)
        elements_added = []
        for _ in range(100):
            element = round(random.random())
            elements_added.append(element)
            buffer.add_element(element)

        sum_value = np.sum(elements_added[-max_buffer_size:])
        assert round(buffer.give_sum_of_elements(), round_to_this_many_decimal_places) == round(sum_value, round_to_this_many_decimal_places), "{}".format(elements_added)
