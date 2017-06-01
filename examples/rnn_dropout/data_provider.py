# -*- coding:utf-8 -*-

from paddle.trainer.PyDataProvider2 import *
import codecs
import random
import cPickle as pkl
import math

UNK_IDX = 2
START = "<s>"
END = "<e>"


def init_word_dict(dict_file):
    """ init word dict. """
    word_dict = {'<s>': 0, '<e>': 1, '<UNK>': 2}
    with codecs.open(dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            lines = line.split('\t')
            word_dict[lines[0]] = int(lines[1]) + 3
    return word_dict


def hook(setting, **kwargs):
    """ hook """

    setting.dict_file = kwargs['dict_file']
    setting.test = kwargs['test']
    setting.use_sampling = False
    if 'nce' in kwargs:
        setting.use_sampling = True
        setting.nce = kwargs['nce']
        setting.sample_num = kwargs['sample_num']

    setting.word_dict = init_word_dict(setting.dict_file)
    dict_size = len(setting.word_dict)
    if setting.use_sampling and not setting.test: 
        setting.slots = [
                integer_value_sequence(dict_size),
                integer_value_sequence(dict_size),
                integer_value_sequence(setting.sample_num + 1) if not setting.nce \
                        else sparse_binary_vector_sequence(setting.sample_num + 1),
                ]
    else:
        setting.slots = [
                integer_value_sequence(dict_size),
                integer_value_sequence(dict_size),
                ]

def _get_ids(word_list, dictionary):
    """ get ids. """
    return [dictionary[START]] + \
           [dictionary.get(word, UNK_IDX) for word in word_list] + \
           [dictionary[END]]

@provider(init_hook=hook)
def process(setting, file_name):
    for line in codecs.open(file_name, 'r', encoding='utf-8'):
        word_list = line.strip().split(' ')
        if not setting.test and len(line) < 2:
            continue
        
        word_ids = [setting.word_dict.get(c, UNK_IDX) for c in word_list]
        word_ids_before = [setting.word_dict[START]] + word_ids
        word_ids_next = word_ids + [setting.word_dict[END]]

        if setting.use_sampling and not setting.test:
            if not setting.nce:
                record = word_ids_before, word_ids_next, [0] * len(word_ids_next)
            else:
                record = word_ids_before, word_ids_next, [[0]] * len(word_ids_next)
        else:
            record = word_ids_before, word_ids_next
        
        yield record
