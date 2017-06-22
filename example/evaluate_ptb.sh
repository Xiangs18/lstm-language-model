#!/bin/bash
source /export/b18/xma/virtual/PyTorch/bin/activate
GPU=`/home/pkoehn/statmt/bin/free-gpu`
LM_PATH='/export/b18/xma/machine_translation/lstm-language-model/'
#echo Using GPU $GPU
CUDA_VISIBLE_DEVICES=$GPU python -u $LM_PATH/evaluate.py \
  --eval_data './data/penn/test.txt' \
  --model 'model/penn-lm.best.pt' \
  --cuda 


