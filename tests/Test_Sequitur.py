from string import ascii_uppercase
import pytest
from utilities.grammar_algorithms.k_Sequitur import k_Sequitur

def test_convert_a_string_using_reverse_rules():
    """Tests convert_a_string_using_reverse_rules method"""
    obj = k_Sequitur(2)
    string = "ababcdeeee"
    rules, reverse_rules, _  = obj.generate_1_layer_of_rules(string)

    new_string = obj.convert_a_string_using_reverse_rules(string, reverse_rules)
    assert new_string == "AAcdBB", new_string

    rules, reverse_rules, _  = obj.generate_1_layer_of_rules(new_string)
    new_string = obj.convert_a_string_using_reverse_rules(new_string, reverse_rules)
    assert new_string == "AAcdBB", new_string

    obj = k_Sequitur(3)
    string = "ababcdeeabeeeeabeeabcd"
    rules, reverse_rules, _  = obj.generate_1_layer_of_rules(string)

    new_string = obj.convert_a_string_using_reverse_rules(string, reverse_rules)
    assert new_string == "AAcdBABBABAcd", new_string

    rules, reverse_rules, _  = obj.generate_1_layer_of_rules(new_string)
    new_string = obj.convert_a_string_using_reverse_rules(new_string, reverse_rules)
    assert new_string == "AAcdCBCCcd", new_string

def test_get_next_rule_name():
    """Tests get_next_rule_name works correctly"""
    obj = k_Sequitur(2)
    string = "aaaabbbb"

    assert obj.rule_names == ascii_uppercase
    assert obj.next_rule_name_ix == 0

    obj.generate_1_layer_of_rules(string)
    assert obj.next_rule_name_ix == 2

    obj.generate_1_layer_of_rules(string)
    assert obj.next_rule_name_ix == 4

    with pytest.raises(ValueError):
        for _ in range(20):
            obj.generate_1_layer_of_rules(string)

def test_generate_1_layer_of_rules_gives_reverse_rules_correctly():
    """Tests generate_1_layer_of_rules"""
    for string in ["agafsfaghghhghghgh", "afdfas", "a", "aabbababababdfdfdfoeorirjajdgsjgkdajfjafdkjasrjajjsjsjsjdjdjkafdkkdksjdfjsdkafklasfkafsl", "aaaaaaaaa"]:
        for k in range(1, 5):
            obj = k_Sequitur(2)
            rules, reverse_rules, _  = obj.generate_1_layer_of_rules(string)
            assert reverse_rules == {v: k for k, v in rules.items()}


def test_generate_1_layer_of_rules():
    """Tests generate_1_layer_of_rules"""
    obj = k_Sequitur(2)
    string = "ababcdeeee"
    rules, reverse_rules, _ = obj.generate_1_layer_of_rules(string)

    assert rules["A"] == "ab"
    assert rules["B"] == "ee"
    assert set(rules.keys()) == {"A", "B"}
    assert reverse_rules["ab"] == "A"
    assert reverse_rules["ee"] == "B"
    assert set(reverse_rules.keys()) == {"ab", "ee"}

    obj = k_Sequitur(3)
    rules, reverse_rules, _  = obj.generate_1_layer_of_rules(string)
    assert set(rules.keys()) == set()
    assert set(reverse_rules.keys()) == set()


    obj = k_Sequitur(2)
    string = "abcdefffgfgfgbcd"
    rules, reverse_rules, _  = obj.generate_1_layer_of_rules(string)
    print(rules)
    assert rules["A"] == "fg"
    assert rules["B"] == "bc"
    assert set(rules.keys()) == {"A", "B"}

def test_generate_grammar():
    """Tests generate_grammar"""
    obj = k_Sequitur(2)
    string = "aababcdaab"
    new_string, all_rules, all_pair_counts = obj.generate_grammar(string)

    assert new_string == "CAcdC"
    assert all_rules == {"A": "ab", "B": "aa", "C": "Bb"}
    assert all_pair_counts == {"ab": 2, "aa": 2, "ba": 1, "cd": 1, "da": 1,
                               "Bb": 2, "bA": 1, "Ac": 1, "dB": 1, "CA": 1, "dC":1}






# from utilities.grammar_algorithms.Sequitur import run_sequitur
#
# def test_Sequitur():
#     assert run_sequitur(
#         'abracadabraabracadabra') == 'Usage\tRule\n 0\tR0 -> R1 R1 \n 2\tR1 -> R2 c a d R2 \n 2\tR2 -> a b r a \n'
#     assert run_sequitur('11111211111') == 'Usage\tRule\n 0\tR0 -> R1 R2 2 R2 R1 \n 3\tR1 -> 1 1 \n 2\tR2 -> R1 1 \n'
#
#     print(run_sequitur('1111'))
#     print('Usage\tRule\n 0\tR0 -> R1 R1 \n 2\tR1 -> 1 1 \n')
#
#
#     assert run_sequitur('1111') == 'Usage\tRule\n 0\tR0 -> R1 R1 \n 2\tR1 -> 1 1\n'