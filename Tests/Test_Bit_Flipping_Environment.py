from Environments.Other_Enrivonments.Bit_Flipping_Environment import Bit_Flipping_Environment


def test_environment_actions():

    env = Bit_Flipping_Environment(5)

    env.state = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]

    env.conduct_action(0)
    env.state = env.next_state
    assert env.state == [0, 0, 0, 1, 0, 1, 0, 0, 1, 0]

    env.conduct_action(0)
    env.state = env.next_state
    assert env.state == [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]

    env.conduct_action(3)
    env.state = env.next_state
    assert env.state == [1, 0, 0, 0, 0, 1, 0, 0, 1, 0]

    env.conduct_action(6)
    env.state = env.next_state
    assert env.state == [1, 0, 0, 0, 0, 1, 0, 0, 1, 0]

def test_environment_goal_achievement():

    env = Bit_Flipping_Environment(5)

    env.state = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    env.desired_goal = [0, 0, 0, 0, 0]

    env.conduct_action(0)
    assert env.reward == -1
    env.state = env.next_state

    assert env.get_achieved_goal() == [0, 0, 0, 1, 0]

    env.conduct_action(2)
    assert env.reward == -1
    env.state = env.next_state

    assert env.get_achieved_goal() == [0, 0, 1, 1, 0]

    env.conduct_action(2)
    assert env.reward == -1
    env.state = env.next_state

    assert env.get_achieved_goal() == [0, 0, 0, 1, 0]

    env.conduct_action(3)
    assert env.reward == 5

