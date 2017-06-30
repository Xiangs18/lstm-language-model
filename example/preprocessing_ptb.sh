#!/bin/bash
source /export/b18/xma/virtual/PyTorch/bin/activate
LM_PATH='..'
python $LM_PATH/preprocess.py \
  --train_data './data/penn/train.txt' \
  --val_data './data/penn/valid.txt' \
  --dict_size 50000 \
  --display_freq 100000 \
  --max_len 100 \
  --trunc_len 100 
