#!/bin/bash
source /export/b18/xma/virtual/PyTorch/bin/activate
GPU=`/home/pkoehn/statmt/bin/free-gpu`
LM_PATH='..'
echo Using GPU $GPU
CUDA_VISIBLE_DEVICES=$GPU python -u $LM_PATH/train.py \
  --train_data './data/penn/train.txt.prep.train.pt' \
  --val_data './data/penn/valid.txt.prep.val.pt' \
  --model_name 'model/exp8-lstm-lm' \
  --dim_word 512 \
  --dim_rnn  1000\
  --num_layers 2 \
  --batch_size 64 \
  --val_batch_size 64 \
  --epoch 10 \
  --optimizer SGD \
  --lr 1 \
  --lr_decay 0.9 \
  --dropout_rate 0.3 \
  --display_freq 100 \
  --save_freq 0\
  --cuda \
  --clip 5 \


