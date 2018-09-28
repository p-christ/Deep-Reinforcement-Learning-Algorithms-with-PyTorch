import random
import numpy as np
from Utilities.Data_Structures.Prioritised_Replay_Buffer import Prioritised_Replay_Buffer

hyperparameters = {
    "alpha_prioritised_replay": 0.5,
    "beta_prioritised_replay": 0.5,
    "incremental_td_error": 0.0
}

def test_prioritised_replay_buffer_add_in_decreasing_elements():

    buffer = Prioritised_Replay_Buffer(max_buffer_size=3, batch_size=None, hyperparameters=hyperparameters)

    buffer.add_experience(100, None, None, None, None, None)

    assert buffer.queue[0].td_error == 100**hyperparameters["alpha"]
    assert buffer.queue[0].heap_index == 1
    assert buffer.heap[1].td_error == 100**hyperparameters["alpha"]

    buffer.add_experience(99, None, None, None, None, None)
    buffer.add_experience(98, None, None, None, None, None)

    assert buffer.queue[1].td_error == 99**hyperparameters["alpha"]
    assert buffer.queue[1].heap_index == 2
    assert buffer.heap[2].td_error == 99**hyperparameters["alpha"]
    assert buffer.queue[2].td_error == 98**hyperparameters["alpha"]
    assert buffer.queue[2].heap_index == 3
    assert buffer.heap[3].td_error == 98**hyperparameters["alpha"]

    buffer.add_experience(97, None, None, None, None, None)

    assert buffer.queue[0].td_error == 97**hyperparameters["alpha"]
    assert buffer.queue[0].heap_index == 2
    assert buffer.heap[2].td_error == 97**hyperparameters["alpha"]
    assert buffer.heap[1].td_error == 99**hyperparameters["alpha"]

    buffer.add_experience(96, None, None, None, None, None)

    assert buffer.queue[1].td_error == 96**hyperparameters["alpha"]
    assert buffer.queue[1].heap_index == 3
    assert buffer.heap[3].td_error == 96**hyperparameters["alpha"]


def test_prioritised_replay_buffer_add_in_increasing_elements():

    buffer = Prioritised_Replay_Buffer(max_buffer_size=3, batch_size=None, hyperparameters=hyperparameters)

    buffer.add_experience(100, None, None, None, None, None)

    assert buffer.queue[0].td_error == 100**hyperparameters["alpha"]
    assert buffer.queue[0].heap_index == 1
    assert buffer.heap[1].td_error == 100**hyperparameters["alpha"]

    buffer.add_experience(101, None, None, None, None, None)
    buffer.add_experience(102, None, None, None, None, None)

    assert buffer.queue[0].td_error == 100**hyperparameters["alpha"]
    assert buffer.queue[1].td_error == 101**hyperparameters["alpha"]
    assert buffer.queue[2].td_error == 102**hyperparameters["alpha"]

    assert buffer.heap[1].td_error == 102**hyperparameters["alpha"]
    assert buffer.heap[2].td_error == 100**hyperparameters["alpha"]
    assert buffer.heap[3].td_error == 101**hyperparameters["alpha"]

    assert buffer.queue[0].heap_index == 2
    assert buffer.queue[1].heap_index == 3
    assert buffer.queue[2].heap_index == 1

    buffer.add_experience(103, None, None, None, None, None)

    assert buffer.queue[0].td_error == 103**hyperparameters["alpha"]

    assert buffer.heap[1].td_error == 103**hyperparameters["alpha"]
    assert buffer.heap[2].td_error == 102**hyperparameters["alpha"]

    assert buffer.queue[0].heap_index == 1
    assert buffer.queue[1].heap_index == 3
    assert buffer.queue[2].heap_index == 2

    buffer.add_experience(104, None, None, None, None, None)

    assert buffer.queue[1].td_error == 104**hyperparameters["alpha"]
    assert buffer.heap[1].td_error == 104**hyperparameters["alpha"]
    assert buffer.heap[3].td_error == 103**hyperparameters["alpha"]

    assert buffer.queue[0].heap_index == 3
    assert buffer.queue[1].heap_index == 1
    assert buffer.queue[2].heap_index == 2


def test_give_max_element_always_keeps_max_at_top():
    max_buffer_size = 50
    for _ in range(1000):
        buffer = Prioritised_Replay_Buffer(max_buffer_size=max_buffer_size, batch_size=None, hyperparameters=hyperparameters)
        elements_added = []
        for _ in range(100):
            element = round(random.random(), 6)
            elements_added.append(element)
            buffer.add_experience(element, None, None, None, None, None)

        max_td_error = np.max(elements_added[-max_buffer_size:])**hyperparameters["alpha"]
        assert buffer.give_max_td_error() == max_td_error, "{}".format(elements_added)

def test_give_sum_of_elements_is_always_correct():
    max_buffer_size = 50
    round_to_this_many_decimal_places = 5
    for _ in range(1000):
        buffer = Prioritised_Replay_Buffer(max_buffer_size=max_buffer_size, batch_size=None, hyperparameters=hyperparameters)
        elements_added = []
        for _ in range(100):
            element = random.random()
            elements_added.append(element)
            buffer.add_experience(element, None, None, None, None, None)

        td_errors_inputted = [(val)**0.5 for val in elements_added[-max_buffer_size:]]
        sum_td_error = np.sum(td_errors_inputted)

        assert round(buffer.give_adapted_sum_of_td_errors(), round_to_this_many_decimal_places) == round(sum_td_error, round_to_this_many_decimal_places), "{}".format(elements_added)

def test_give_num_layers_in_heap():

    buffer = Prioritised_Replay_Buffer(max_buffer_size=10, batch_size=None, hyperparameters=hyperparameters)

    buffer.add_experience(1, None, None, None, None, None)

    assert buffer.give_num_layers_in_heap() == 1

    buffer.add_experience(2, None, None, None, None, None)
    assert buffer.give_num_layers_in_heap() == 2

    buffer.add_experience(3, None, None, None, None, None)
    assert buffer.give_num_layers_in_heap() == 2

    buffer.add_experience(4, None, None, None, None, None)
    assert buffer.give_num_layers_in_heap() == 3
    buffer.add_experience(5, None, None, None, None, None)
    assert buffer.give_num_layers_in_heap() == 3

    buffer.add_experience(6, None, None, None, None, None)
    buffer.add_experience(7, None, None, None, None, None)
    assert buffer.give_num_layers_in_heap() == 3

    buffer.add_experience(8, None, None, None, None, None)
    assert buffer.give_num_layers_in_heap() == 4


def test_give_sample_indexes():
    batch_size = 5
    buffer = Prioritised_Replay_Buffer(max_buffer_size=8, batch_size=batch_size, hyperparameters=hyperparameters)

    for _ in range(100):

        buffer.add_experience(1, None, None, None, None, None)
        buffer.add_experience(1, None, None, None, None, None)
        buffer.add_experience(1, None, None, None, None, None)
        buffer.add_experience(1, None, None, None, None, None)
        buffer.add_experience(1, None, None, None, None, None)
        buffer.add_experience(1, None, None, None, None, None)
        buffer.add_experience(1, None, None, None, None, None)
        buffer.add_experience(1, None, None, None, None, None)

        _, indexes = buffer.pick_experiences_based_on_rank()

        assert len(indexes) == batch_size





