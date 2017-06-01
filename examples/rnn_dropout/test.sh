#!/bin/bash

set -e

cfg=trainer_config_sampling.py
output_dir=output_is
start_pass=0

env LOG_logtostderr=1 
env GLOG_minloglevel=0 
env GLOG_log_dir="log" 
paddle train \
  --job=test \
  --config=$cfg \
  --save_dir=${output_dir} \
  --test_pass=${start_pass} \
  --trainer_count=5 \
  --use_gpu=false \
  --config_args test=1 \
  2>&1 | tee 'test.log'

