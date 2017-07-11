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
emb_size = 24
max_id = 3

q = data_layer(name='q', size=max_id)
k = data_layer(name='k', size=max_id)
v = data_layer(name='v', size=max_id)
target = data_layer(name='target', size=max_id)

emb_para = ParamAttr('_embedding_rand',
 is_static=False
 )

q_emb = embedding_layer(name='q_emb', input=q, size=emb_size,
                    param_attr=emb_para)
k_emb = embedding_layer(name='k_emb', input=k, size=emb_size,
                    param_attr=emb_para)
v_emb = embedding_layer(name='v_emb', input=v, size=emb_size,
                    param_attr=emb_para)

def multihead_attention(ipt, num_heads, head_size, prefix, kv_ipt=None, mask=None):
    head_atts = []
    if kv_ipt is None:
        kv_ipt = ipt
    for i in xrange(num_heads):
        q = fc_layer(ipt, name=prefix + 'q_%d' % i, size=head_size, act=IdentityActivation())
        k = fc_layer(kv_ipt, name=prefix + 'k_%d' % i, size=head_size, act=IdentityActivation())
        v = fc_layer(kv_ipt, name=prefix + 'v_%d' % i, size=head_size, act=IdentityActivation())
        scale_dot = scale_dot_att_layer(q, k, v, scale_strategy='sqrt_k', mask_strategy=mask)
        head_atts.append(scale_dot)
    
    concat = concat_layer(head_atts)
    return fc_layer(concat, size=num_heads * head_size, act=ReluActivation())

def residual_fn(x, y, drop_rate):
    # TODO: correct dropout config
    #y = dropout_layer(y, dropout_rate=drop_rate, share_dropout_mask_in_seq=False)
    z = x + y
    return batch_norm_layer(input=z)


num_heads = 4
head_size = 6
drop_rate = 0.1
def encoder_layer(ipt, prefix):
    att = multihead_attention(ipt, num_heads, head_size, prefix)
    att = residual_fn(ipt, att, drop_rate)
    att_fc = fc_layer(input=att, size=att.size,
    #layer_attr=ExtraAttr(drop_rate=drop_rate)
    )
    return residual_fn(att, att_fc, drop_rate)

encode_layer_num = 3
encoder_out = q_emb
for i in xrange(encode_layer_num):
    prefix = 'encoder_%d' % i
    encoder_out = encoder_layer(encoder_out, prefix)
print_layer(encoder_out)

# decoder
def decoder_layer(decode_ipt, encode_out, prefix, mask=None):
    last_step = multihead_attention(decode_ipt, num_heads, head_size, prefix + '_decode_att')
    last_step = residual_fn(decode_ipt, last_step, drop_rate)

    att = multihead_attention(last_step, num_heads, head_size, prefix + '_encode_decode_att',
     kv_ipt=encode_out, mask=mask)
    att = residual_fn(last_step, att, drop_rate)

    att_fc = fc_layer(input=att, size=att.size)
    return residual_fn(att, att_fc, drop_rate)


decode_layer_num = 3
decoder_out = k_emb
for i in xrange(decode_layer_num):
    prefix = 'decoder_%d' % i
    mask = "default_mask" if i == 0 else None
    decoder_out = decoder_layer(decoder_out, encoder_out, prefix, mask=mask)

print_layer(decoder_out)
prediction = fc_layer(decoder_out, size=max_id, act=SoftmaxActivation())
#print_layer(q_emb)
#print_layer(k_emb)

#with mixed_layer(name='dot_out', size=1,
#        #act=SequenceSoftmaxActivation()
#        ) as dot_out:
#    #dot_out += dotmul_operator(a=q_emb, b=k_emb)
#    dot_out += seqmul_operator(a=q_emb, b=k_emb)
#print_layer(dot_out)
#fake2 = fc_layer(input=dot_out, size=max_id, act=SoftmaxActivation())
#for name in [q_emb, k_emb, v_emb]:
#    print_layer(name)
#fake2 = scale_dot_att_layer(q_emb, k_emb, v_emb)
#print_layer(fake2)

#fake_out = fc_layer(input=k_emb, size=max_id, act=SoftmaxActivation())
cls = classification_cost(input=prediction, label=target)
sum_evaluator(input=cls)

inputs(q, k, v, target)
outputs(cls)
