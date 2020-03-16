#!/bin/bash

python src/train_nri.py \
  --var 1e-4 --prediction_steps 10 --epochs 100 --lr-decay 50\
  --discrim_epochs 15 --discrim_hidden_size 512 --gan_epochs 500 \
  --discrim_lr 1e-3 \
  --gan_lr 2e-4 \
  --train_gan \
  --ignore_pretrain \
   # --load_folder exp_nri/pretrain
