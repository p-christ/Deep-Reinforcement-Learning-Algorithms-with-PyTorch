from Other_Enrivonments.Bit_Flipping_Environment import Bit_Flipping_Environment


def test_environment_actions():

    env = Bit_Flipping_Environment(5)

    env.state = [1, 0, 0, 1, 0]

    env.conduct_action(0)
    env.state = env.next_state
    assert env.state == [0, 0, 0, 1, 0]

    env.conduct_action(0)
    env.state = env.next_state
    assert env.state == [1, 0, 0, 1, 0]

    print(env.state)

