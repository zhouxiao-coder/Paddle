# -*- encoding:utf-8 -*-

from paddle.trainer_config_helpers import *
from data_provider import init_word_dict
import codecs
import cPickle as pkl

test = get_config_arg('test', bool, False)

data_file = "train.list"
test_file = "test.list"
dict_file = "dict.txt"

define_py_data_sources2(train_list=data_file,
                        test_list=test_file,
                        module="data_provider",
                        obj="process",
                        args={"dict_file": dict_file, "test": test})

################## Algorithm Config #####################
settings(
    batch_size=20,
    learning_method=AdaDeltaOptimizer(),  # origin optimizer
    learning_rate=1.0,
    regularization=L2Regularization(2e-3),
    gradient_clipping_threshold=5,
)

default_initial_std(5e-3)  # from 5e-3
default_initial_strategy(1)
#################### Network Config ######################
w_para = ParamAttr(name='fc.w')
b_para = ParamAttr(name='fc.b')
hid_size = 300
word_dim = 9999 + 3

words = data_layer(name='word', size=word_dim)
emb = embedding_layer(input=words, size=hid_size,
                      param_attr=ParamAttr('_embedding', learning_rate=2.0))
lstm0 = simple_lstm(input=emb, size=hid_size)
decoder_input = lstm0

true_label = data_layer(name='true_label', size=word_dim)
output = fc_layer(input=decoder_input, param_attr=w_para, bias_attr=b_para,
                    size=word_dim, act=SoftmaxActivation())

cls = classification_cost(input=output, label=true_label)
sum_evaluator(input=cls)

inputs(words, true_label)
outputs(cls)
