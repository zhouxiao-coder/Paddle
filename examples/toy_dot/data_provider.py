# -*- coding:utf-8 -*-
'''
'''
from paddle.trainer.PyDataProvider2 import *
import codecs
import random
import cPickle as pkl
import math
import random

max_id = 10

def hook(setting, **kwargs):
    """ hook """
    # src_words, src_pos, trg_words, trg_pos, trg_next
    setting.slots = [
        integer_value_sequence(max_id),
        integer_value_sequence(max_id),
        integer_value_sequence(max_id),
        integer_value_sequence(max_id),
        integer_value_sequence(max_id),
        #integer_value_sequence(max_id),
        ]


@provider(init_hook=hook)
def process(setting, file_name):
    """ provider function """
    src_words = [1, 2, 3, 4]
    trg_words = [2, 3, 4, 5, 6]
    pad = 0
    def positions(l):
        return list(range(l))

    for i in xrange(10):
        yield [
            src_words,
            positions(len(src_words)),
            [0] + trg_words,
            positions(len(trg_words) + 1),
            trg_words + [max_id - 1]
        ]
        # yield [[0, 1, 2, 0], [0, 1, 0, 2], [0, 0, 1, 2], [0, 1, 0, 0],
        # [0, 1, 2, 0],
        # ]
