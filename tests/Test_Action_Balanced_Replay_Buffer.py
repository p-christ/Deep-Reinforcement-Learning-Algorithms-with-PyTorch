import random
from collections import Counter

import pytest

from utilities.data_structures.Action_Balanced_Replay_Buffer import Action_Balanced_Replay_Buffer

def test_add_experience():
    """Tests that add_experience works correctly"""
    buffer = Action_Balanced_Replay_Buffer(6, 4, 0, 3)

    rewards = [0 for _ in range(4)]
    next_states = [0 for _ in range(4)]
    states = [0 for _ in range(4)]
    dones = [0 for _ in range(4)]
    actions = [0, 1, 2, 0]

    for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        buffer.add_experience(state, action, reward, next_state, done)

    assert len(buffer.memories[0]) == 2
    assert len(buffer.memories[1]) == 1
    assert len(buffer.memories[2]) == 1

    buffer.add_experience(99, 0, 0, 0, 0)
    assert len(buffer.memories[0]) == 2
    assert buffer.memories[0][1].state == 99

    buffer = Action_Balanced_Replay_Buffer(6, 4, 0, 3)
    buffer.add_experience(states, actions, rewards, next_states, dones)
    assert len(buffer.memories[0]) == 2
    assert len(buffer.memories[1]) == 1
    assert len(buffer.memories[2]) == 1

    buffer.add_experience(99, 0, 0, 0, 0)
    assert len(buffer.memories[0]) == 2
    assert buffer.memories[0][1].state == 99

def test_add_experience_throws_error():
    """Tests that add_experience works correctly"""
    buffer = Action_Balanced_Replay_Buffer(20, 4, 0, 3)
    with pytest.raises(KeyError):
        buffer.add_experience(3, 99, 1, 0, 0)
        buffer.sample()

    buffer = Action_Balanced_Replay_Buffer(20, 4, 0, 3)
    buffer.add_experience(3, 2, 1, 0, 0)

    with pytest.raises(AssertionError):
        buffer.sample()

def test_sample_correctly():
    """Tests that sample works correctly"""
    buffer = Action_Balanced_Replay_Buffer(20, 4, 0, 3)
    buffer.add_experience(3, 2, 1, 0, 0)
    buffer.add_experience(2, 0, 1, 0, 0)
    buffer.add_experience(1, 1, 1, 0, 0)
    states, actions, rewards, next_states, dones = buffer.sample()

    for var in [states, actions, rewards, next_states, dones]:
        assert len(var) == 4

    num_occurances = 0
    tries = 50

    for random_seed in range(tries):
        buffer = Action_Balanced_Replay_Buffer(20, 4, random_seed, 3)
        buffer.add_experience(3, 2, 1, 0, 0)
        buffer.add_experience(2, 0, 1, 0, 0)
        buffer.add_experience(1, 1, 1, 0, 0)
        states, actions, rewards, next_states, dones = buffer.sample()
        if states[2] == 3.0: num_occurances += 1
        print(states)
    assert num_occurances < tries/2
    assert num_occurances > tries/5

def test_sample_statistics_correct():
    """Tests that sampled experiences correspond to expected statistics"""
    tries = 5
    for random_seed in range(tries):
        for num_actions in range(1, 7):
            for buffer_size in [random.randint(55, 9999) for _ in range(10)]:
                for batch_size in [random.randint(8, 200) for _ in range(10)]:
                    buffer = Action_Balanced_Replay_Buffer(buffer_size, batch_size, random.randint(0, 2000000), num_actions)
                    for _ in range(500):
                        random_action = random.randint(0, num_actions - 1)
                        buffer.add_experience(1, random_action, 1, 0, 0)
                    states, actions, rewards, next_states, dones = buffer.sample()
                    actions = [action.item() for action in actions]
                    assert len(actions) == batch_size
                    count = Counter(actions)
                    action_count = count[0]
                    for action in range(num_actions):
                        assert abs(count[action] - action_count) < 2, print(count[action])




