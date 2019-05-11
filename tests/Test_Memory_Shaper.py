from Memory_Shaper import Memory_Shaper

memory_shaper = Memory_Shaper(20, 5, 2)

def test_calculate_max_action_length():
    """Tests that calculate_max_action_length works correctly"""
    action_rules = {(0, 2, 33, 1, 22, 0, 0): 99, (0, 4): 2, (0, 9): 100}
    assert memory_shaper.calculate_max_action_length(action_rules) == 7

    action_rules = {(0, 2, 3): 99, (0, 4, 0, 0): 2, (0, 9): 100}
    assert memory_shaper.calculate_max_action_length(action_rules) == 4


