#!/bin/bash
source /export/b18/xma/virtual/PyTorch/bin/activate
GPU=`/home/pkoehn/statmt/bin/free-gpu`
LM_PATH='/export/b18/xma/machine_translation/lstm-language-model/'
#echo Using GPU $GPU
CUDA_VISIBLE_DEVICES=$GPU python -u $LM_PATH/main.py \
  --train_data './data/exp8/train.bpe.en.prep.train.pt' \
  --val_data './data/exp8/dev.bpe.en.prep.val.pt' \
  --model_name 'model/exp8-lstm-lm' \
  --dim_word 256 \
  --dim_rnn  256\
  --num_layers 2 \
  --batch_size 64 \
  --val_batch_size 64 \
  --epoch 10 \
  --model_reload \
  --optimizer sgd \
  --lr 1 \
  --lr_decay 0.9 \
  --dropout_rate 0.3 \
  --display_freq 1000\
  --save_freq 20000\
  --cuda \
  --clip 5 \


