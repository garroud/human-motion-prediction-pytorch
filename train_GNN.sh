#!/bin/bash


# test using GNN for training
python src/run_human_motion.py --residual_velocities --use_GNN \
--iterations 150000 \
--num_passing 2 --batch_size 8 \
--seq_length_out 10 \
--size 256 \
--learning_rate 5e-4 --save_every 1000 --test_every 500 \
--show_every 100 \
--do_prob 0.2 \
