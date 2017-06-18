#!/bin/bash
source /export/b18/xma/virtual/PyTorch/bin/activate
GPU=`/home/pkoehn/statmt/bin/free-gpu`
CUDA_VISIBLE_DEVICES=$GPU python main.py \
  --train_data './data/penn/train.txt' \
  --val_data './data/penn/valid.txt' \
  --model_name 'model/LSTM_LM'\
  --dim_word 256\
  --dim_rnn 512\
  --num_layers 1 \
  --batch_size 64 \
  --val_batch_size 64 \
  --epoch 10 \
  --data_reload \
  --model_reload \
  --optimizer sgd \
  --lr 0.1 \
  --dropout_rate 0.3 \
  --display_freq 100 \
  --save_freq 10 \
  --cuda \


