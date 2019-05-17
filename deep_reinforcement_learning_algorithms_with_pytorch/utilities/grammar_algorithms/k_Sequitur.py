
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

class k_Sequitur(object):

    def __init__(self, k, end_of_episode_symbol="/"):
        self.k = k
        self.end_of_episode_symbol = end_of_episode_symbol
        self.next_rule_name_ix = 0

    def generate_action_grammar(self, actions):
        """Generates a grammar given a list of actions"""
        assert isinstance(actions, list)
        assert len(actions) > 0, "Need to provide a list of at least 1 action"
        assert isinstance(actions[0], int), "The actions should be integers"
        new_actions, all_rules, rule_usage = self.discover_all_rules_and_new_actions_representation(actions)

        print("New actions ", new_actions)
        print("Rule usage ", rule_usage)

        action_usage = self.extract_action_usage_from_rule_usage(rule_usage, all_rules)
        return new_actions, all_rules, action_usage

    def discover_all_rules_and_new_actions_representation(self, actions):
        """Takes in a list of actions and discovers all the rules present that get used more than self.k times and the
        subsequent new actions list when all rules are applied recursively"""
        all_rules = {}
        current_actions = None
        new_actions = actions
        rule_usage = defaultdict(int)
        print("Step 1")
        while new_actions != current_actions:
            current_actions = new_actions
            print("Current actions ", current_actions)
            rules, reverse_rules = self.generate_1_layer_of_rules(current_actions)
            print("Rules generated here ", rules)
            all_rules.update(rules)
            new_actions, rules_usage_count = self.convert_a_string_using_reverse_rules(current_actions, reverse_rules)
            print("Rules usage count here ", rules_usage_count)
            for key in rules_usage_count.keys():
                rule_usage[key] += rules_usage_count[key]
        return new_actions, all_rules, rule_usage

    def generate_1_layer_of_rules(self, string):
        """Generate dictionaries indicating the pair of symbols that appear next to each other more than self.k times"""
        pairs_of_symbols = defaultdict(int)
        last_pair = None
        skip_next_symbol = False
        rules = {}

        for ix in range(len(string) - 1):
            # We skip the next symbol if it is already being used in a rule we just made
            if skip_next_symbol:
                skip_next_symbol = False
                continue
            # We skip this symbol if the next one is the end of the episode
            if string[ix+1] == self.end_of_episode_symbol: continue

            pair = (string[ix], string[ix+1])
            # We don't count a pair if it was the previous pair (and therefore we have 3 of the same symbols in a row)
            if pair != last_pair:
                pairs_of_symbols[pair] += 1
                last_pair = pair
            else: last_pair = None
            if pairs_of_symbols[pair] >= self.k:
                previous_pair = (string[ix-1], string[ix])
                pairs_of_symbols[previous_pair] -= 1
                skip_next_symbol = True
                if pair not in rules.values():
                    rule_name = self.get_next_rule_name()
                    rules[rule_name] = pair
        reverse_rules = {v: k for k, v in rules.items()}
        return rules, reverse_rules

    def get_next_rule_name(self):
        """Returns next rule name to use and increments count """
        next_rule_name = "R{}".format(self.next_rule_name_ix)
        self.next_rule_name_ix += 1
        return next_rule_name

    def convert_symbol_to_raw_actions(self, symbol, rules):
        """Converts a symbol back to the sequence of raw actions it represents"""
        assert not isinstance(symbol, list)
        assert isinstance(symbol, str) or isinstance(symbol, int)
        symbol = [symbol]
        finished = False
        while not finished:
            new_symbol = []
            for symbol_val in symbol:
                if symbol_val in rules.keys():
                    new_symbol.append(rules[symbol_val][0])
                    new_symbol.append(rules[symbol_val][1])
                else:
                    new_symbol.append(symbol_val)
            if new_symbol == symbol: finished = True
            else: symbol = new_symbol
        new_symbol = tuple(new_symbol)
        return new_symbol

    def extract_action_usage_from_rule_usage(self, rule_usage, all_rules):
        """Extracts the usage of each action (of 2 or more primitive actions) out from the usage of each rule"""
        action_usage = {}
        for key in rule_usage.keys():
            action_usage[self.convert_symbol_to_raw_actions(key, all_rules)] = rule_usage[key]
        return action_usage

    def convert_a_string_using_reverse_rules(self, string, reverse_rules):
        """Converts a string using the rules we have previously generated"""
        new_string = []
        skip_next_element = False
        rules_usage_count = defaultdict(int)
        for ix in range(len(string)):
            if skip_next_element:
                skip_next_element = False
                continue
            # If is last element in string and wasn't just part of a pair then we add it to new string and finish
            if ix == len(string) - 1:
                new_string.append(string[ix])
                continue
            pair = (string[ix], string[ix+1])
            if pair in reverse_rules.keys():
                result = reverse_rules[pair]
                rules_usage_count[result] += 1
                new_string.append(result)
                skip_next_element = True
            else:
                new_string.append(string[ix])
        return new_string, rules_usage_count

