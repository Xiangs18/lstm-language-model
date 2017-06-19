#!/bin/bash
source /export/b18/xma/virtual/PyTorch/bin/activate
GPU=`/home/pkoehn/statmt/bin/free-gpu`
echo Using GPU $GPU
CUDA_VISIBLE_DEVICES=$GPU python main.py \
  --train_data './data/penn/train.txt' \
  --val_data './data/penn/valid.txt' \
  --model_name 'model/LSTM_LM'\
  --dim_word 200\
  --dim_rnn 200\
  --num_layers 2 \
  --batch_size 64 \
  --val_batch_size 64 \
  --epoch 2 \
  --model_reload \
  --optimizer sgd \
  --lr 5 \
  --dropout_rate 0.2 \
  --display_freq 100\
  --save_freq 1 \
  --cuda \
  --clip 5 \


