from utilities.data_structures.Prioritised_Replay_Buffer import Prioritised_Replay_Buffer
import numpy as np
import random

hyperparameters = {
        "alpha_prioritised_replay": 0.5,
        "beta_prioritised_replay": 0.5,
        "incremental_td_error": 0.0,
        "buffer_size": 4,
        "batch_size": 3
}


def test_prioritised_replay_buffer():

    buffer = Prioritised_Replay_Buffer(hyperparameters)
    buffer.add_experience(100, 1, 2, 3, 4, 5)

    assert buffer.deque[0].key == 100.0**hyperparameters["alpha_prioritised_replay"]
    assert buffer.deque[0].value == (1, 2, 3, 4, 5)
    assert buffer.deque[0].heap_index == 1
    assert buffer.heap[1].key == 100.0**hyperparameters["alpha_prioritised_replay"]
    assert buffer.heap[1].value == (1, 2, 3, 4, 5)

    buffer.add_experience(99, 1, 2, 3, 4, 5)
    buffer.add_experience(98, 1, 2, 3, 4, 5)

    assert buffer.deque[0].key == 100.0**hyperparameters["alpha_prioritised_replay"]
    assert buffer.deque[0].value == (1, 2, 3, 4, 5)
    assert buffer.deque[0].heap_index == 1
    assert buffer.heap[1].key == 100.0**hyperparameters["alpha_prioritised_replay"]
    assert buffer.heap[1].value == (1, 2, 3, 4, 5)

    assert buffer.deque[1].key == 99.0**hyperparameters["alpha_prioritised_replay"]
    assert buffer.deque[1].value == (1, 2, 3, 4, 5)
    assert buffer.deque[1].heap_index == 2
    assert buffer.heap[2].key == 99.0**hyperparameters["alpha_prioritised_replay"]
    assert buffer.heap[2].value == (1, 2, 3, 4, 5)

    assert buffer.deque[2].key == 98.0**hyperparameters["alpha_prioritised_replay"]
    assert buffer.deque[2].value == (1, 2, 3, 4, 5)
    assert buffer.deque[2].heap_index == 3
    assert buffer.heap[3].key == 98.0**hyperparameters["alpha_prioritised_replay"]
    assert buffer.heap[3].value == (1, 2, 3, 4, 5)

    buffer.add_experience(105, 1, 2, 3, 4, 5)

    assert buffer.deque[3].key == 105.0**hyperparameters["alpha_prioritised_replay"]
    assert buffer.deque[3].value == (1, 2, 3, 4, 5)
    assert buffer.deque[3].heap_index == 1
    assert buffer.heap[1].key == 105.0**hyperparameters["alpha_prioritised_replay"]
    assert buffer.heap[1].value == (1, 2, 3, 4, 5)
    assert buffer.heap[2].key == 100.0 ** hyperparameters["alpha_prioritised_replay"]

    buffer.add_experience(101, 1, 24, 3, 4, 5)

    assert buffer.deque[0].key == 101.0 ** hyperparameters["alpha_prioritised_replay"]
    assert buffer.deque[0].value == (1, 24, 3, 4, 5)
    assert buffer.deque[0].heap_index == 2
    assert buffer.heap[2].key == 101.0 ** hyperparameters["alpha_prioritised_replay"]
    assert buffer.heap[2].value == (1, 24, 3, 4, 5)


def test_heap_always_keeps_max_element_at_top():
    hyperparameters["buffer_size"] = 200
    for _ in range(100):
        buffer = Prioritised_Replay_Buffer(hyperparameters)
        elements_added = []
        for ix in range(1, 100):
            element = random.random()
            elements_added.append(element)
            buffer.add_experience(element, 0, 0, 0, 0, 0)

        max_key = np.max(elements_added)** hyperparameters["alpha_prioritised_replay"]
        assert round(buffer.give_max_td_error(), 8) == round(max_key, 8), "{}".format(elements_added)

def test_give_sum_of_elements_is_always_correct():
    hyperparameters["buffer_size"] = 200
    for _ in range(100):
        buffer = Prioritised_Replay_Buffer(hyperparameters)
        elements_added = []
        for ix in range(1, 100):
            element = random.random()
            elements_added.append((abs(element) + hyperparameters["incremental_td_error"]) ** hyperparameters["alpha_prioritised_replay"])
            buffer.add_experience(element, 0, 0, 0, 0, 0)

            sum_key = np.sum(elements_added)
        assert round(buffer.give_adapted_sum_of_td_errors(), 8) == round(sum_key, 8), "{}".format(elements_added)
