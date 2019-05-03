from environments.Four_Rooms_Environment import Four_Rooms_Environment
from random import randint
from collections import Counter


def test_location_to_state():
    """Tests location_to_state maps each location to a unique integer"""
    for num_rows in [12, 10]:
        for num_cols in [15, 9]:
            env = Four_Rooms_Environment(grid_width=num_cols, grid_height=num_rows)
            observed_states = set()
            for row in range(num_rows):
                for col in range(num_cols):
                    state = env.location_to_state((row, col))
                    assert state not in observed_states
                    observed_states.add(state)

def test_actions_execute_correctly():
    """Tests that actions execute correctly"""
    env = Four_Rooms_Environment(stochastic_actions_probability=0.0)
    env.reset()
    env.move_user(env.current_user_location, (3, 3))

    env.step(0)
    assert env.current_user_location == (2, 3)

    env.step(1)
    assert env.current_user_location == (2, 4)

    env.step(2)
    assert env.current_user_location == (3, 4)

    env.step(3)
    assert env.current_user_location == (3, 3)

    env.step(0)
    assert env.current_user_location == (2, 3)

    env.step(0)
    assert env.current_user_location == (1, 3)

    env.step(0)
    assert env.current_user_location == (1, 3)

    env.step(1)
    assert env.current_user_location == (1, 4)

    env.step(1)
    assert env.current_user_location == (1, 5)

    env.step(1)
    assert env.current_user_location == (1, 5)

def test_check_user_location_and_goal_location_match_state_and_next_state():
    """Checks whether user location always matches state and next state correctly"""
    for _ in range(50):
        env = Four_Rooms_Environment()
        env.reset()
        for _ in range(50):
            move = randint(0, 3)
            env.step(move)
            assert env.state == [env.location_to_state(env.current_user_location), env.location_to_state(env.current_goal_location)]
            assert env.next_state == [env.location_to_state(env.current_user_location), env.location_to_state(env.current_goal_location)]

def test_lands_on_goal_correctly():
    """Checks whether getting to goal state produces the correct response"""
    env = Four_Rooms_Environment(stochastic_actions_probability=0.0)
    env.reset()
    env.move_user(env.current_user_location, (3, 3))
    env.move_goal(env.current_goal_location, (2, 2))

    env.step(0)
    assert env.reward == env.step_reward_for_not_achieving_goal
    assert not env.done

    env.step(3)
    assert env.reward == env.reward_for_achieving_goal
    assert env.done

    env = Four_Rooms_Environment(stochastic_actions_probability=0.0)
    env.reset()
    env.move_user(env.current_user_location, (2, 3))
    env.move_goal(env.current_goal_location, (2, 8))
    for move in [2, 1, 1, 1, 1, 1, 0]:
        env.step(move)
        if move != 0:
            assert env.reward == env.step_reward_for_not_achieving_goal
            assert not env.done
        else:
            assert env.reward == env.reward_for_achieving_goal
            assert env.done

def test_location_to_state_and_state_to_location_match():
    """Test that location_to_state and state_to_location are inverses of each other"""
    env = Four_Rooms_Environment(stochastic_actions_probability=0.0)
    env.reset()
    for row in range(env.grid_height):
        for col in range(env.grid_width):
            assert env.location_to_state((row, col)) == env.location_to_state(env.state_to_location(env.location_to_state((row, col))))

def test_randomness_of_moves():
    """Test that determine_which_action_will_actually_occur correctly implements stochastic_actions_probability"""
    env = Four_Rooms_Environment(stochastic_actions_probability=0.0)
    env.reset()
    for _ in range(10):
        for move in env.actions:
            assert move == env.determine_which_action_will_actually_occur(move)

    env = Four_Rooms_Environment(stochastic_actions_probability=1.0)
    num_iterations = 10000
    for move in env.actions:
        moves = []
        for _ in range(num_iterations):
            moves.append(env.determine_which_action_will_actually_occur(move))
        count = Counter(moves)
        for move_test in env.actions:
            if move != move_test: #We do this because stochastic probability 1.0 means the move will never be picked
                assert abs((num_iterations / (len(env.actions)-1)) - count[move_test]) < num_iterations / 20.0,  "{}".format(count)

    env = Four_Rooms_Environment(stochastic_actions_probability=0.75)
    num_iterations = 10000
    for move in env.actions:
        moves = []
        for _ in range(num_iterations):
            moves.append(env.determine_which_action_will_actually_occur(move))
        count = Counter(moves)
        for move_test in env.actions:
            assert abs((num_iterations / len(env.actions)) - count[move_test]) < num_iterations / 20.0, "{}".format(count)


