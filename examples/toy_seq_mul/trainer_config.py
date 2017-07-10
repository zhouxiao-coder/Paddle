# -*- encoding:utf-8 -*-

from paddle.trainer_config_helpers import *
import codecs
import cPickle as pkl

test = False
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
    batch_size=2,
    learning_method=MomentumOptimizer(),  # origin optimizer
    learning_rate=1.0,
    regularization=L2Regularization(2e-3),
    gradient_clipping_threshold=5,
)

#################### Network Config ######################
emb_size = 2
max_id = 3

q = data_layer(name='q', size=max_id)
k = data_layer(name='k', size=max_id)
v = data_layer(name='v', size=max_id)
target = data_layer(name='target', size=max_id)

emb_para = ParamAttr('_embedding', is_static=True)

q_emb = embedding_layer(name='q_emb', input=q, size=emb_size,
                    param_attr=emb_para)
k_emb = embedding_layer(name='k_emb', input=k, size=emb_size,
                    param_attr=emb_para)
v_emb = embedding_layer(name='v_emb', input=v, size=emb_size,
                    param_attr=emb_para)
#print_layer(q_emb)
#print_layer(k_emb)

with mixed_layer(name='dot_out', size=1,
        #act=SequenceSoftmaxActivation()
        ) as dot_out:
    #dot_out += dotmul_operator(a=q_emb, b=k_emb)
    dot_out += seqmul_operator(a=q_emb, b=k_emb)
print_layer(dot_out)
fake2 = fc_layer(input=dot_out, size=max_id, act=SoftmaxActivation())

fake_out = fc_layer(input=k_emb, size=max_id, act=SoftmaxActivation())
cls = classification_cost(input=fake_out, label=target)
sum_evaluator(input=cls)

inputs(q, k, target)
outputs(cls)
