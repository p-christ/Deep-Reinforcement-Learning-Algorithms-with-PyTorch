from environments.Bit_Flipping_Environment import Bit_Flipping_Environment
import numpy as np


def test_environment_actions():
    """Tests environment is executing actions correctly"""
    env = Bit_Flipping_Environment(5)
    env.reset()
    env.state = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]

    env.step(0)
    env.state = env.next_state
    assert env.state == [0, 0, 0, 1, 0, 1, 0, 0, 1, 0]

    env.step(0)
    env.state = env.next_state
    assert env.state == [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]

    env.step(3)
    env.state = env.next_state
    assert env.state == [1, 0, 0, 0, 0, 1, 0, 0, 1, 0]

    env.step(6)
    env.state = env.next_state
    assert env.state == [1, 0, 0, 0, 0, 1, 0, 0, 1, 0]

def test_environment_goal_achievement():
    """Tests environment is registering goal achievement properly"""
    env = Bit_Flipping_Environment(5)
    env.reset()
    env.state = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    env.desired_goal = [0, 0, 0, 0, 0]

    env.step(0)
    assert env.reward == -1
    env.state = env.next_state
    assert env.achieved_goal == [0, 0, 0, 1, 0]

    env.step(2)
    assert env.reward == -1
    env.state = env.next_state
    assert env.achieved_goal == [0, 0, 1, 1, 0]

    env.step(2)
    assert env.reward == -1
    env.state = env.next_state
    assert env.achieved_goal == [0, 0, 0, 1, 0]

    env.step(3)
    assert env.reward == 5

def test_compute_reward():
    """Tests compute_reward method"""
    env = Bit_Flipping_Environment(5)
    assert env.compute_reward(np.array([0, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 0]), None) == env.reward_for_achieving_goal
    assert env.compute_reward(np.array([1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 1]), None) == env.reward_for_achieving_goal
    assert env.compute_reward(np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0]), None) == env.reward_for_achieving_goal
    assert env.compute_reward(np.array([1, 1, 1, 1, 1]), np.array([0, 0, 0, 1, 0]), None) == env.step_reward_for_not_achieving_goal
    assert env.compute_reward(np.array([1, 1, 1, 1, 1]), np.array([0, 0, 0, 0, 0]), None) == env.step_reward_for_not_achieving_goal

