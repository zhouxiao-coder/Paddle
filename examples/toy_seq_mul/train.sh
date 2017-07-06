#!/bin/bash

set -e

cfg=trainer_config.py
flag=full
nce=0
subtract_log_q=0
output_dir=output_${flag}

env LOG_logtostderr=1 
env GLOG_minloglevel=0 
#env GLOG_log_dir="log" 
paddle train \
  --config=$cfg \
  --save_dir=${output_dir} \
  --trainer_count=1 \
  --log_period=10 \
  --num_passes=1 \
  --use_gpu=false \
  --show_parameter_stats_period=10 \
  --dot_period=5 \
  --test_period=1000 \
  --config_args test=0,nce=$nce,subtract_log_q=${subtract_log_q},sample_num=25\
  --init_model_path=./ \
  --load_missing_parameter_strategy zero \
  2>&1 | tee train_${flag}.log
