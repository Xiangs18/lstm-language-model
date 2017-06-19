#!/bin/bash -v

qsub -l 'hostname=b1[123456789]*|c*,gpu=1' \
  -cwd \
  -o ./exp-log/train.`date +"%Y-%m-%d.%H-%M-%S"`.out \
  -e ./exp-log/train.`date +"%Y-%m-%d.%H-%M-%S"`.err \
  ./train.sh

