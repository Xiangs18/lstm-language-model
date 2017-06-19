#!/bin/bash
LM_PATH='/export/b18/xma/machine_translation/lstm-language-model'
python $LM_PATH/preprocess.py \
  --train_data './data/exp8/train.bpe.en' \
  --val_data './data/exp8/dev.bpe.en' \
  --dict_size 50000 \
  --display_freq 100000 \
