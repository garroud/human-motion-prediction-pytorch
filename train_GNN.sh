

# test using GNN for training
python src/translate.py --residual_velocities --use_GNN \
--iterations 150000 \
--num_passing 2 --batch_size 8 \
--seq_length_out 10 \
--size 256 \
--stochastic --irl_training \
--learning_rate 5e-4 --save_every 1000 --test_every 500 \
--train_discrim_iter 20000 \
--train_GAN_iter 50000  \
--show_every 100 \
--do_prob 0.2 \
--load 39000
# --skip_pretrain_policy \
# --test
# --sample --load 50000
