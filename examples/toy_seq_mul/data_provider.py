# -*- coding:utf-8 -*-
'''
'''
from paddle.trainer.PyDataProvider2 import *
import codecs
import random
import cPickle as pkl
import math
import random

max_id = 3

def hook(setting, **kwargs):
    """ hook """
    # src_words, src_pos, trg_words, trg_pos
    setting.slots = [
        integer_value_sequence(max_id),
        integer_value_sequence(max_id),
        integer_value_sequence(max_id),
        integer_value_sequence(max_id)]


@provider(init_hook=hook)
def process(setting, file_name):
    """ provider function """
    for i in xrange(10):
        yield [[0, 1, 2, 0], [0, 1, 0, 2], [0, 0, 1, 2], [0, 1, 0, 0]]
