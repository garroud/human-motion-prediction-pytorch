#!/bin/bash

#Train the model

NOW=$(date +"%Y%m%d%H%M%S")
# python src/translate.py --learning_rate 0.005 --residual_velocities --num_layers 3 2>&1 | tee logs/log-$NOW.txt



# #experiments for the write up.
# python src/translate.py --iterations 50000 --learning_rate 0.005 --residual_velocities --save_every 2000 --test_every 2000 --stochastic --irl_training \
#  --skip_pretrain_policy

# for sample
python src/translate.py --iterations 50000 --learning_rate 0.005 --residual_velocities --save_every 2000 --test_every 2000 --stochastic --irl_training \
--skip_pretrain_policy --sample --load 20000
# python src/translate.py --learning_rate 0.005 --residual_velocities --architecture basic 2>&1 | tee logs/untied.txt
# python src/translate.py --learning_rate 0.005 --residual_velocities
# python src/translate.py --iterations 50000 --learning_rate 0.005 --residual_velocities --stochastic --irl_training --sample --load 50000
# python src/translate.py --learning_rate 0.005 --residual_velocities --omit_one_hot 2>&1 | tee logs/upsup.txt


# for action in walking eating smoking discussion directions greeting phoning posing purchases sitting sittingdown takingphoto waiting walkingdog walkingtogether
# do
# python src/translate.py --action ${action} --learning_rate 0.05 --residual_velocities -save_every 10000 2>&1 | tee "logs/residual_SA_${action}.txt"
# python src/translate.py --action ${action} --learning_rate 0.05 --residual_velocities -save_every 10000 2>&1 | tee "logs/residual_SA_${action}.txt"

# python src/translate.py --action ${action} --learning_rate 0.05 --seq_length_out 25 --save_every 10000 2>&1 | tee "logs/sample_SA_${action}.txt"
# python src/translate.py --action ${action} --learning_rate 0.05 --seq_length_out 25 --save_every 10000 2>&1 | tee "logs/sample_SA_${action}.txt"
# done
# for action in walking eating smoking discussion directions greeting phoning posing purchases sitting sittingdown takingphoto waiting walkingdog walkingtogether
# do
# python src/forward_kinematics.py --sample_name samples_tf.h5 --action_name "${action}_0" --save --save_name "figs/${action}_tf.gif"
# python src/forward_kinematics.py --sample_name samples_tf.h5 --action_name "${action}_0"

# python src/forward_kinematics.py --sample_name samples.h5 --action_name "${action}_0" --save --save_name "figs/${action}_py.gif"
# python src/forward_kinematics.py --sample_name samples.h5 --action_name "${action}_0"
# convert "figs/${action}_py.gif" "cvtpng/${action}_py.png"
# done
