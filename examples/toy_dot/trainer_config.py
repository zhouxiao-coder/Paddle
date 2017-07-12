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
max_id = 10
num_heads = 4
head_size = 6
drop_rate = 0.1

src_word = data_layer(name='src_word', size=max_id)
src_pos = data_layer(name='src_pos', size=max_id)
trg_word = data_layer(name='trg_word', size=max_id)
trg_pos = data_layer(name='trg_pos', size=max_id)
trg_next_word = data_layer(name='trg_next_word', size=max_id)
#trg_next_pos = data_layer(name='trg_next_pos', size=max_id)

emb_para = ParamAttr('_embedding_mid',
 is_static=True
 )

pos_emb_para = ParamAttr('_pos_embedding_mid',
 is_static=True
 )

src_emb = embedding_layer(name='src_emb', input=src_word, size=emb_size,
                    param_attr=emb_para)
src_pos_emb = embedding_layer(name='src_pos_emb', input=src_pos, size=emb_size,
                    param_attr=pos_emb_para)
trg_emb = embedding_layer(name='trg_emb', input=trg_word, size=emb_size,
                    param_attr=emb_para)
trg_pos_emb = embedding_layer(name='trg_pos_emb', input=trg_pos, size=emb_size,
                    param_attr=pos_emb_para)

trg_next_emb = embedding_layer(name='trg_next_emb', input=trg_next_word, size=emb_size,
                    param_attr=emb_para)
# trg_next_pos_emb = embedding_layer(name='trg_next_pos_emb', input=trg_next_pos, size=emb_size,
#                     param_attr=pos_emb_para)

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
    y = dropout_layer(y, dropout_rate=drop_rate, share_dropout_mask_in_seq=False)
    z = x + y
    return batch_norm_layer(input=z)

def encoder_layer(ipt, prefix):
    att = multihead_attention(ipt, num_heads, head_size, prefix)
    att = residual_fn(ipt, att, drop_rate)
    att_fc = fc_layer(input=att, size=att.size,
    layer_attr=ExtraAttr(drop_rate=drop_rate)
    )
    return residual_fn(att, att_fc, drop_rate)

encode_layer_num = 1
#encoder_out = q_emb
encoder_out = addto_layer(input=[src_emb, src_pos_emb])
for i in xrange(encode_layer_num):
    prefix = 'encoder_%d' % i
    encoder_out = encoder_layer(encoder_out, prefix)
print_layer(encoder_out)

# decoder
def decoder_layer(decode_ipt, encode_out, prefix, mask=None):
    last_step = multihead_attention(decode_ipt, num_heads, head_size, prefix + '_decode_att', mask=mask)
    last_step = residual_fn(decode_ipt, last_step, drop_rate)

    att = multihead_attention(last_step, num_heads, head_size, prefix + '_encode_decode_att', 
    kv_ipt=encode_out, mask=None)
    att = residual_fn(last_step, att, drop_rate)

    att_fc = fc_layer(input=att, size=att.size)
    return residual_fn(att, att_fc, drop_rate)


decode_layer_num = 1
#decoder_out = k_emb
decoder_out = addto_layer(input=[trg_emb, trg_pos_emb])
for i in xrange(decode_layer_num):
    prefix = 'decoder_%d' % i
    mask = "default_mask" if i == 0 else None
    decoder_out = decoder_layer(decoder_out, encoder_out, prefix, mask=mask)

print_layer(decoder_out)
prediction = fc_layer(decoder_out, size=max_id, act=SoftmaxActivation())
cls = classification_cost(input=prediction, label=trg_next_word)
sum_evaluator(input=cls)

inputs(src_word, src_pos, trg_word, trg_pos, trg_next_word)
outputs(cls)
