#!/bin/bash

set -e

flag=full
output_dir=output_${flag}

env LOG_logtostderr=1 
env GLOG_minloglevel=0 
#env GLOG_log_dir="log" 
paddle train \
  --config=$cfg \
  --save_dir=${output_dir} \
  --trainer_count=4 \
  --log_period=20 \
  --num_passes=100 \
  --use_gpu=false \
  --show_parameter_stats_period=500 \
  --dot_period=5 \
  --test_period=1000 \
  --config_args test=0 \
  2>&1 | tee train_${flag}.log
