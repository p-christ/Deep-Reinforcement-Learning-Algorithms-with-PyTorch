

# NOTE that this is not my code and it was taken from https://github.com/craignm/sequitur/blob/master/python/sequitur.py


# Sequitur by Craig-Nevill Manning
# Ported from Java code by Eibe Frank
# Ported by Ravi Annaswamy 2 hours on May 8, 2013
# Not fully tested yet :)


class Symbol:
    def __init__(self):
        self.num_terminals = 100000
        self.prime = 2265539
        self.value = ' '
        global digrams

    def clone(self):
        sym = Symbol()
        sym.value = self.value
        sym.n = self.n
        sym.p = self.p
        return sym

    def join(self, left, right):
        if left.n != None:
            left.delete_digram()
            if (right.p != None and right.n != None and
                    right.value == right.p.value and
                    right.value == right.n.value):
                digrams[str(right.value) + str(right.n.value)] = right
            if (left.p != None and left.n != None and
                    left.value == left.p.value and
                    left.value == left.n.value):
                digrams[str(left.p.value) + str(left.value)] = left.p
        left.n = right
        right.p = left

    def insert_after(self, to_insert):
        self.join(to_insert, self.n)
        self.join(self, to_insert)

    def digram(self):
        return str(self.value) + str(self.n.value)

    def delete_digram(self):
        if self.n.is_guard(): return

        if self.digram() in digrams:
            dummy = digrams[self.digram()]
            if dummy == self:
                del digrams[self.digram()]

    def is_guard(self):
        return False

    def is_nonterminal(self):
        return False

    def check(self):
        if self.n.is_guard():
            return False

        if self.digram() not in digrams:
            digrams[self.digram()] = self
            return False

        found = digrams[self.digram()]
        if found.n != self:
            self.match(self, found)
        return True

    def cleanup(self):
        pass

    def substitute(self, r):
        self.cleanup()
        self.n.cleanup()
        self.p.insert_after(NonTerminal(r))
        if not self.p.check(): self.p.n.check()

    def match(self, newD, matching):
        # print 'Matching ', newD.value, matching.value
        global numRules

        if matching.p.is_guard() and matching.n.n.is_guard():
            # print 'Using Rule'
            r = (matching.p).r
            newD.substitute(r)
        else:
            # print 'Creating Rule'
            r = Rule(numRules)
            numRules += 1
            first = newD.clone()
            second = newD.n.clone()
            r.the_guard.n = first
            first.p = r.the_guard
            first.n = second
            second.p = first
            second.n = r.the_guard
            r.the_guard.p = second
            matching.substitute(r)
            newD.substitute(r)
            digrams[str(first.value) + str(first.n.value)] = first

        # print 'Checking Rule Utility'
        if (r.first()).is_nonterminal() and (r.first()).r.count == 1:
            (r.first()).expand()

    def hash_code(self):
        code = 21599 * self.value + 20507 * self.n.value
        code = code % self.prime
        return code

    def equals(self, obj):
        return self.value == obj.value and self.n.value == obj.n.value


class Guard(Symbol):
    def __init__(self, rule):
        self.r = rule
        self.value = ' '
        self.p = self
        self.n = self

    def cleanup(self):
        self.join(self.p, self.n)

    def is_guard(self):
        return True

    def check(self):
        return False

class Rule:

    def __init__(self, numRules):
        self.number = numRules
        self.the_guard = Guard(self)
        self.count = 0
        self.index = 0

    def first(self):
        return self.the_guard.n

    def last(self):
        return self.the_guard.p

    def get_rules(self):
        rules = []
        processedRules = 0
        text = ''
        charCounter = 0
        text += "Usage\tRule\n"
        rules.append(self)

        while (processedRules < len(rules)):
            currentRule = rules[processedRules]
            text += " " + str(currentRule.count) + '\tR' + str(processedRules) + ' -> '
            sym = currentRule.first()

            while True:
                if sym.is_guard():
                    break
                if sym.is_nonterminal():
                    rule = sym.r
                    if len(rules) > rule.index and rules[rule.index] == rule:
                        index = rule.index
                    else:
                        index = len(rules)
                        rule.index = index
                        rules.append(rule)
                    text += 'R' + str(index)
                else:
                    if sym.value == ' ':
                        text += '_'
                    elif sym.value == '\n':
                        text += '\\n'
                    else:
                        text += str(sym.value)
                text += ' '
                sym = sym.n
            text += '\n'
            processedRules += 1
        return text


digrams = {}



class Terminal(Symbol):
    def __init__(self, value):
        self.value = value
        self.p = None
        self.n = None

    def cleanup(self):
        self.join(self.p, self.n)
        self.delete_digram()

    def clone(self):
        sym = Terminal(self.value)
        sym.p = self.p
        sym.n = self.n
        return sym


class NonTerminal(Symbol):
    def __init__(self, rule):
        self.r = rule
        self.r.count += 1
        # self.value=self.numTerminals+self.r.number
        self.value = self.r.number
        self.p = None
        self.n = None

    def clone(self):
        sym = NonTerminal(self.r)
        sym.p = self.p
        sym.n = self.n
        return sym

    def cleanup(self):
        self.join(self.p, self.n)
        self.delete_digram()
        self.r.count -= 1

    def is_nonterminal(self):
        return True

    def expand(self):
        self.join(self.p, self.r.first())
        self.join(self.r.last(), self.n)
        digrams[str(self.r.last().value) + str(self.r.last().n.value)] = self.r.last()
        self.r.the_guard.r = None
        self.r.the_guard = None





numRules = 0


def run_sequitur(text):
    global numRules
    first_rule = Rule(numRules)
    numRules += 1

    global digrams
    digrams = {}

    for c in text:
        (first_rule.last()).insert_after(Terminal(c))
        first_rule.last().p.check()

    return first_rule.get_rules()

# print(run_sequitur('abracadabraabracadabra'))

#
if __name__ == "__main__":
    print(run_sequitur('abracadabraabracadabra'))
    print(run_sequitur('11111211111'))


#
# def test():
#     assert run_sequitur(
#         'abracadabraabracadabra') == 'Usage\tRule\n 0\tR0 -> R1 R1 \n 2\tR1 -> R2 c a d R2 \n 2\tR2 -> a b r a \n'
#     assert run_sequitur('11111211111') == 'Usage\tRule\n 0\tR0 -> R1 R2 2 R2 R1 \n 3\tR1 -> 1 1 \n 2\tR2 -> R1 1 \n'

#
# test()

# test()