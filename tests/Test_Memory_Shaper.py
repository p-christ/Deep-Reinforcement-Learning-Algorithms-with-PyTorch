from Memory_Shaper import Memory_Shaper
import numpy as np

buffer_size = 10
batch_size = 5
seed = 1


def new_reward_fn(cumulative_reward, length_of_macro_action, reward_increment):
    if cumulative_reward == 0.0: increment = 0.1
    else: increment = abs(cumulative_reward)
    cumulative_reward += increment * ((length_of_macro_action - 1)** 0.5) * reward_increment
    return cumulative_reward


def test_calculate_max_action_length():
    """Tests that calculate_max_action_length works correctly"""

    memory_shaper = Memory_Shaper(buffer_size, batch_size, seed, new_reward_fn=new_reward_fn)
    action_rules = {(0, 2, 33, 1, 22, 0, 0): 99, (0, 4): 2, (0, 9): 100}
    assert memory_shaper.calculate_max_action_length(action_rules) == 7

    action_rules = {(0, 2, 3): 99, (0, 4, 0, 0): 2, (0, 9): 100}
    assert memory_shaper.calculate_max_action_length(action_rules) == 4

def test_add_adapted_experience_for_an_episode():
    """Tests that add_adapted_experience_for_an_episode works correctly"""
    for reward_increment in [0.0, 0.5, 1.5]:
        buffer_size = 3
        memory_shaper = Memory_Shaper(buffer_size, buffer_size, seed,
                                      new_reward_fn=lambda x, y: x + reward_increment * ((y - 1)** 0.5) * 0.1,
                                      action_balanced_replay_buffer=False)
        memory_shaper.reset()
        states = [0, 1]
        next_states = [1, 10]
        rewards = [10, 5]
        actions = [0, 5]
        dones = [False, True]
        memory_shaper.add_episode_experience(states, next_states, rewards, actions, dones)

        action_rules = {6:(0, 5), 1: (1,), 2:(2,), 3:(3,), 4:(4,), 5:(5,), 0:(0,)}

        replay_buffer = memory_shaper.put_adapted_experiences_in_a_replay_buffer(action_rules)

        assert len(replay_buffer) == 3

        s_states, s_actions, s_rewards, s_next_states, s_dones = replay_buffer.sample(separate_out_data_types=True)

        print(s_rewards)
        print(new_reward_fn(15.0, 2, reward_increment))

        assert all(s_states.numpy() == np.array([[0.0], [1.0,], [0.0]]))
        assert all(s_actions.numpy() == np.array([[0.0], [5.0, ], [6.0]]))
        assert all(s_rewards.numpy() == np.array([[10.0], [5.0, ], [new_reward_fn(15.0, 2, reward_increment)]]))
        assert all(s_next_states.numpy() == np.array([[1.0], [10.0, ], [10.0]]))
        assert all(s_dones.numpy() == np.array([[0.0], [1.0, ], [1.0]]))

        buffer_size = 5
        memory_shaper = Memory_Shaper(buffer_size, buffer_size, seed, new_reward_fn=lambda x, y: x + reward_increment * ((y - 1) ** 0.5) * 0.1,
                                      action_balanced_replay_buffer=False)
        memory_shaper.reset()
        states = [0, 1, 2]
        next_states = [1, 10, 11]
        rewards = [10, 5, -4]
        actions = [0, 5, 2]
        dones = [False, False, True]
        memory_shaper.add_episode_experience(states, next_states, rewards, actions, dones)

        action_rules = {6: (0, 5), 7: (0, 5, 2), 1: (1,), 2:(2,), 3:(3,), 4:(4,), 5:(5,), 0:(0,)}

        replay_buffer = memory_shaper.put_adapted_experiences_in_a_replay_buffer(action_rules)

        assert len(replay_buffer) == 5

        s_states, s_actions, s_rewards, s_next_states, s_dones = replay_buffer.sample(
            separate_out_data_types=True)

        print(s_rewards)

        assert all(s_states.numpy() == np.array([[0.0], [0.0], [1.0], [0.0], [2.0]]))
        assert all(s_actions.numpy() == np.array([[6.0], [0.0], [5.0], [7.0], [2.0]]))
        assert all(s_rewards.numpy() == np.array([[new_reward_fn(15.0, 3, reward_increment)], [10.0], [5.0], [new_reward_fn(11.0, 2, reward_increment)], [-4.0]]))
        assert all(s_next_states.numpy() == np.array([[10.0], [1.0], [10.0], [11.0], [11.0]]))
        assert all(s_dones.numpy() == np.array([[0.0], [0.0], [0.0], [1.0], [1.0]]))

def test_add_adapted_experience_for_an_episode_long_action_length():
    """Tests that add_adapted_experience_for_an_episode works correctly for actions with length > 2"""
    for reward_increment in [0.0, 0.5, 1.5]:
        buffer_size = 4
        memory_shaper = Memory_Shaper(buffer_size, buffer_size, seed, new_reward_fn=new_reward_fn)
        states = [0, 1, 2]
        next_states = [1, 10, 11]
        rewards = [10, 5, 2]
        actions = [0, 7, 3]
        dones = [False, False, False]
        memory_shaper.add_episode_experience(states, next_states, rewards, actions, dones)

        action_rules = {(0, 7, 3): 8}

        replay_buffer = memory_shaper.put_adapted_experiences_in_a_replay_buffer(action_rules)

        assert len(replay_buffer) == 4

        s_states, s_actions, s_rewards, s_next_states, s_dones = replay_buffer.sample(separate_out_data_types=True)

        assert all(s_states.numpy() == np.array([[2.0], [0.0,], [1.0], [0.0]]))
        assert all(s_actions.numpy() == np.array([[3.0], [0.0, ], [7.0], [8.0]]))
        assert all(s_rewards.numpy() == np.array([[2.0], [10.0], [5.0], [17.0*(1.0 + reward_increment)]]))
        assert all(s_next_states.numpy() == np.array([[11.0], [1.0, ], [10.0], [11.0]]))
        assert all(s_dones.numpy() == np.array([[0.0], [0.0, ], [0.0], [0.0]]))


def test_add_adapted_experience_for_multiple_episodes():
    """Tests that add_adapted_experience_for_an_episode works correctly for multiple episodes"""
    for reward_increment in [0.0, 0.5, 1.5]:
        buffer_size = 2

        memory_shaper = Memory_Shaper(buffer_size, buffer_size, seed, new_reward_fn)
        states = [0]
        next_states = [1]
        rewards = [10]
        actions = [0]
        dones = [False]
        memory_shaper.add_episode_experience(states, next_states, rewards, actions, dones)

        states = [1]
        next_states = [2]
        rewards = [11]
        actions = [1]
        dones = [True]
        memory_shaper.add_episode_experience(states, next_states, rewards, actions, dones)

        action_rules = {}

        replay_buffer = memory_shaper.put_adapted_experiences_in_a_replay_buffer(action_rules)

        assert len(replay_buffer) == 2

        s_states, s_actions, s_rewards, s_next_states, s_dones = replay_buffer.sample(separate_out_data_types=True)

        assert all(s_states.numpy() == np.array([[0.0], [1.0]]))
        assert all(s_actions.numpy() == np.array([[0.0], [1.0]]))
        assert all(s_rewards.numpy() == np.array([[10.0], [11.0]]))
        assert all(s_next_states.numpy() == np.array([[1.0], [2.0]]))
        assert all(s_dones.numpy() == np.array([[0.0], [1.0]]))





