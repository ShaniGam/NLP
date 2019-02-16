from collections import defaultdict
import getopt
import random


class PCFG(object):
    def __init__(self):
        self._rules = defaultdict(list)
        self._sums = defaultdict(float)
        self.tree_str = ''
        self.sent = []

    def add_rule(self, lhs, rhs, weight):
        assert(isinstance(lhs, str))
        assert(isinstance(rhs, list))
        self._rules[lhs].append((rhs, weight))
        self._sums[lhs] += weight

    @classmethod
    def from_file(cls, filename):
        grammar = PCFG()
        with open(filename) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if not line: continue
                w,l,r = line.split(None, 2)
                r = r.split()
                w = float(w)
                grammar.add_rule(l,r,w)
        return grammar

    def is_terminal(self, symbol): return symbol not in self._rules

    def gen(self, symbol):
        if self.is_terminal(symbol):
            self.tree_str += " " + symbol
            self.sent.append(symbol)
        else:
            self.tree_str += ' (' + symbol
            expansion = self.random_expansion(symbol)
            for s in expansion:
                self.gen(s)
            self.tree_str += ")"

    def random_sent(self):
        self.tree_str = ""
        self.sent = []
        self.gen("ROOT")
        first_word = self.sent[0]
        first_word_upper = first_word[0].upper() + first_word[1:]
        self.sent = self.sent[1:]
        self.sent = [first_word_upper] + self.sent
        sentence = ' '.join(self.sent)
        self.tree_str = self.tree_str[1:]
        return sentence

    def random_expansion(self, symbol):
        """
        Generates a random RHS for symbol, in proportion to the weights.
        """
        p = random.random() * self._sums[symbol]
        for r,w in self._rules[symbol]:
            p = p - w
            if p < 0: return r
        return r

    def get_tree(self):
        return self.tree_str

if __name__ == '__main__':

    import sys

    pcfg = PCFG.from_file(sys.argv[1])
    num_of_sen = 1
    create_tree = False
    try:
        opts, args = getopt.getopt(sys.argv[2:], "tn:", ["num="])
    except getopt.GetoptError:
        print "error"
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-n":
            num_of_sen = int(arg)
        elif opt == "-t":
            create_tree = True

    for i in range(num_of_sen):
        print pcfg.random_sent()
        if create_tree:
            print pcfg.get_tree()

