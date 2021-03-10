import random
import math
import textdistance
from collections import namedtuple
import copy


PatternNode = namedtuple('PatternNode', ['pattern', 'chars', 'weight'])
TextStats = namedtuple('TextStats', ['mean', 'std'])
EditStatSet = namedtuple('EditStatSet', ['char', 'clss', 'group'])

CONVERSION_CACHE = {}

ALPHA = set(list('abcdefghijklmnopqrstuvwxyz'))
NUMERIC = set(list('1234567890'))
# potentially add the "\t" character
WHITESPACE = set([' '])
# purposefully remove [", ;] as they have trouble getting escaped when
# writing spec files
SYMBOLS = set(list('<,>.?/:\'{[}]|\_-+=}!@#$%^&*()'))
OTHER = set()
OTHER_SYMBOL = 'O'

CLASS_MAPPINGS = {
    'A': ALPHA,
    'D': NUMERIC,
    'O': OTHER,
    'S': SYMBOLS,
    'W': WHITESPACE
}



def get_class(char):
    '''returns the string for the character representing the class
    '''

    conversion = CONVERSION_CACHE.get(char)
    if conversion is not None:
        return conversion

    for symbol, charclass in CLASS_MAPPINGS.items():
        if char.lower() in charclass:
            CONVERSION_CACHE[char] = symbol
            return symbol
    # no matches, use "OTHER" class
    CONVERSION_CACHE[char] = OTHER_SYMBOL
    return OTHER_SYMBOL


def do_randomly(chance=0.5):
    return random.random() > chance

OP_ADD = 'add'
OP_EDIT = 'edit'
OP_REMOVE = 'remove'
OPS = [OP_ADD, OP_EDIT, OP_REMOVE]

def random_op():
    '''return a random op type
    '''
    return OPS[round(random.random() * (len(OPS) - 1))]

class RString():
    def __init__(self, real):
        '''Converts a string into character class representation

        e.g. if you have a string "123 Main St", this would be converted
        into an uncompressed list of regex classes [dddwaaawaaa]. (d for digit, w for whitespace, a for alphabetic, o for other)
        '''
        self.orig = real
        self.chars = ''.join([get_class(c) for c in real])
        self.groups = []

        last_idx = 0
        cur_cnt = 0
        cur_group = None
        for char in self.chars:
            if cur_group != char:
                if cur_group is not None:
                    self.groups.append(PatternNode(cur_group, self.orig[last_idx:last_idx + cur_cnt], cur_cnt,))
                    last_idx += cur_cnt
                cur_group = char
                cur_cnt = 0
            cur_cnt += 1

        self.groups.append(PatternNode(cur_group, self.orig[last_idx:len(self.orig)], cur_cnt,))
        self.groupstr = ''.join([x.pattern for x in self.groups])

    def _compute_distance(self, x, y):
        # print("computing distance of {} vs {}".format(x, y))
        return textdistance.damerau_levenshtein(x, y)

    def class_distance(self, other):
        return self._compute_distance(self.chars, other.chars)

    def orig_distance(self, other):
        return self._compute_distance(self.orig, other.orig)

    def group_distance(self, other):
        return self._compute_distance(self.groupstr, other.groupstr)

    def generate_mutation(self, stat_set, edit_push=1):
        '''generates a mutation of the original string based on the stat set
        '''
        newgroups = copy.deepcopy(self.groups)
        def rand_group():
            return round(random.random()*(len(self.groups) - 1))
        # choose N group to edit/delete/insert
        # groups_to_edit = round(random.gauss(stat_set.group.mean, stat_set.group.std))
        # if groups_to_edit > 1:
        #     # pick random group to edit
        #     for group in range(0, groups_to_edit):
        #         group_to_edit = rand_group()
        #         op = random_op()

        # # choose N class to edit/delete/insert
        # classes_to_edit = round(random.gauss(stat_set.clss.mean, stat_set.clss.std))
        # if classes_to_edit > 1:
        #     pass


        # choose N characters to edit/delete/insert
        chars_to_edit = round(random.expovariate(1/float(stat_set.char.mean - 1))) + 1
        print("editing {} chars".format(chars_to_edit))
        if chars_to_edit > 1:
            for char in range(0, chars_to_edit):
                # loop in case group needs to be picked multiple times (e.g. edited group now becomes empty)
                while True:
                    try:
                        op = random_op()
                        group = rand_group()
                        egrp = newgroups[group]
                        if op == OP_ADD:
                            # generate character in group class
                            charcls = CLASS_MAPPINGS[egrp.pattern]
                            random_char = random.sample(list(charcls), 1)[0]
                            # put character in random location in the group
                            txt = egrp.chars
                            loc = random.sample(range(len(txt)), 1)[0]
                            txt = txt[:loc] + random_char + txt[loc:]
                            newgroups[group] = PatternNode(egrp.pattern, txt, egrp.weight)
                            break

                        if op == OP_EDIT:
                            if len(egrp.chars) == 1:
                                # don't edit groups with 1 char
                                continue
                            # generate character in group class
                            charcls = CLASS_MAPPINGS[egrp.pattern]
                            random_char = random.sample(list(charcls), 1)[0]
                            # replace character in random location in the group
                            txt = list(egrp.chars)
                            loc = random.sample(range(len(txt)), 1)[0]
                            txt[loc] = random_char
                            txt = "".join(txt)
                            newgroups[group] = PatternNode(egrp.pattern, txt, egrp.weight)
                            break

                        if op == OP_REMOVE:
                            if len(egrp.chars) == 1:
                                # don't remove groups with 1 char
                                continue
                            # replace character in random location in the group
                            txt = list(egrp.chars)
                            loc = random.sample(range(len(txt)), 1)[0]
                            del txt[loc]
                            txt = "".join(txt)
                            newgroups[group] = PatternNode(egrp.pattern, txt, egrp.weight)
                            break
                    except Exception as e:
                        pass

        newstr = []
        for grp in newgroups:
            newstr.append(grp.chars)

        return "".join(newstr)




    def __str__(self):
        return '{}'.format(self.groups)