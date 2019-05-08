
# My implementation of the k_Sequitur algorithm described in the papers: https://arxiv.org/pdf/cs/9709102.pdf
# and https://www.biorxiv.org/content/biorxiv/early/2018/03/13/281543.full.pdf
# The algorithm takes in a sequence and forms a grammar using two rules:
# 1) No pair of adjacent symbols appears more than k times in the grammar
# 2) Every rule in the grammar is used more than k times
#
# e.g. string "abddddeabde" with k=2 would turn to:
# "AdddBAB"
# R1: A --> ab
# R2: B --> de
from collections import defaultdict
from string import ascii_uppercase



class k_Sequitur(object):

    def __init__(self, k):
        self.k = k
        self.rule_names = ascii_uppercase
        self.next_rule_name_ix = 0

    def generate_grammar(self, string):
        assert isinstance(string, str), "Need to provide 1 long string"
        assert len(string) > 0, "Need to provide a string of at least 1 character"

        all_rules = {}
        all_pair_counts = {}
        current_string = None
        new_string = string

        while new_string != current_string:
            current_string = new_string
            rules, reverse_rules, pairs_of_symbols = self.generate_1_layer_of_rules(current_string)
            all_pair_counts.update(pairs_of_symbols)
            all_rules.update(rules)
            print(rules)
            new_string = self.convert_a_string_using_reverse_rules(current_string, reverse_rules)

        return new_string, all_rules, all_pair_counts


    def generate_1_layer_of_rules(self, string):


        pairs_of_symbols = defaultdict(int)
        last_pair = None
        skip_next_symbol = False
        rules = {}

        for ix in range(len(string) - 1):

            # We skip the next symbol if it is already being used in a rule we just made
            if skip_next_symbol:
                skip_next_symbol = False
                continue

            pair = string[ix] + string[ix+1]

            # We don't count a pair if it was the previous pair (and therefore we have 3 of the same symbols in a row)
            if pair != last_pair:
                pairs_of_symbols[pair] += 1
                last_pair = pair
            else:
                last_pair = None
            if pairs_of_symbols[pair] >= self.k:
                skip_next_symbol = True
                if pair not in rules.values():
                    rule_name = self.get_next_rule_name()
                    rules[rule_name] = pair
        reverse_rules = {v: k for k, v in rules.items()}
        return rules, reverse_rules, pairs_of_symbols

    def convert_a_string_using_reverse_rules(self, string, reverse_rules):
        """Converts a string using the rules we have previously generated"""
        new_string = []
        skip_next_element = False

        for ix in range(len(string)):
            if skip_next_element:
                skip_next_element = False
                continue

            # If is last element in string and wasn't just part of a pair then we add it to new string and finish
            if ix == len(string) - 1:
                new_string.append(string[ix])
                continue

            pair = string[ix] + string[ix+1]
            if pair in reverse_rules.keys():
                result = reverse_rules[pair]
                new_string.append(result)
                skip_next_element = True
            else:
                new_string.append(string[ix])

        new_string = "".join(new_string)

        return new_string

    def get_next_rule_name(self):
        """Returns next rule name to use and increments count """
        next_rule_name = self.rule_names[self.next_rule_name_ix]
        self.next_rule_name_ix += 1
        if self.next_rule_name_ix >= len(self.rule_names): raise ValueError("Ran out of rule names")
        return next_rule_name




obj = k_Sequitur(2)
string = "aaaabaaaab"

new_string, rules, all_pair_counts = obj.generate_grammar(string)

print("Rules ", rules)
print("new string ", new_string)
print("pair counts", all_pair_counts)