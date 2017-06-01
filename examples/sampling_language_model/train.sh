#!/bin/bash

set -e

case "$1" in
  "full")
    cfg=trainer_config.py
    flag=full
    nce=0
    subtract_log_q=0
    ;;
  "is")
    cfg=trainer_config_sampling.py
    flag=is
    nce=0
    subtract_log_q=1
    ;;
  "nce")
    cfg=trainer_config_sampling.py
    flag=nce
    nce=1
    subtract_log_q=2
    ;;
  *)
    echo "Please use one of {full|is|nce} to specify model type"
    exit 1
esac
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
  --config_args test=0,nce=$nce,subtract_log_q=${subtract_log_q},sample_num=25\
  2>&1 | tee train_${flag}.log
