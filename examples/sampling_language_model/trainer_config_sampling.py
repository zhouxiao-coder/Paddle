# -*- encoding:utf-8 -*-

from paddle.trainer_config_helpers import *
from data_provider import init_word_dict
import codecs
import cPickle as pkl

test = get_config_arg('test', bool, False)
nce = get_config_arg('nce', bool, False)
subtract_log_q = get_config_arg('subtract_log_q', int, 1)
sample_num = get_config_arg('sample_num', int, 25)

data_file = "train.list"
test_file = "test.list"
dict_file = "dict.txt"
sample_num = 25

define_py_data_sources2(train_list=data_file,
                        test_list=test_file,
                        module="data_provider",
                        obj="process",
                        args={"dict_file": dict_file, "sample_num": sample_num,
                         "nce": nce, "test": test})

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
# load noise distribution, which is unigram distribution raised to power 0.75
import cPickle as pkl
import math

pkl_file = '../data/ptb.prob.pkl'
word_distr = pkl.load(open(pkl_file))

hid_size = 300
word_dim = 9999 + 3
w_para = ParamAttr(name='fc.w')
if nce:
    # initialize std to -log(||V||)
    b_para = ParamAttr(name='fc.b', initial_mean=-math.log(word_dim),
                       initial_std=0.001)
else:
    b_para = ParamAttr(name='fc.b')

words = data_layer(name='word', size=word_dim)
emb = embedding_layer(input=words, size=hid_size,
                      param_attr=ParamAttr('_embedding', learning_rate=2.0))
lstm0 = simple_lstm(input=emb, size=hid_size)
decoder_input = lstm0

true_label = data_layer(name='true_label', size=word_dim)
if test:
    output = selective_fc_layer(input=lstm0,
                                param_attr=w_para,
                                bias_attr=b_para,
                                select=None,
                                has_selected_colums=False,
                                pass_generation=False,
                                size=word_dim,
                                act=SoftmaxActivation(),
                                name='output'
                                )

    cls = classification_cost(input=output, label=true_label)
    sum_evaluator(input=cls)

    inputs(words, true_label)
    outputs(output)
else:
    # fake label used for sampling, index position 0 is always the true value
    label = data_layer(name='label', size=sample_num + 1)
    exps = sampled_fc_layer(
        input=lstm0,
        label=true_label,
        param_attr=w_para,
        bias_attr=b_para,
        full_output_size=word_dim,
        sampled_output_size=sample_num + 1,
        num_neg_samples=sample_num,
        neg_sampling_dist=word_distr,
        subtract_log_q=subtract_log_q,
        share_sample_in_batch=False,
        act=SoftmaxActivation() if not nce else SigmoidActivation(),
    )

    if not nce:
        cls = classification_cost(input=exps, label=label)
        sum_evaluator(input=cls)
    else:
        cls = multi_binary_label_cross_entropy(input=exps, label=label)
        sum_evaluator(input=cls)

    inputs(words, true_label, label)
    outputs(cls)
