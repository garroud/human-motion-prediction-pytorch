#!/bin/bash

for action in walking eating smoking discussion directions greeting phoning posing purchases sitting sittingdown takingphoto waiting walkingdog walkingtogether
do

python src/forward_kinematics.py --sample_name samples.h5 --action_name "${action}_0" --save --save_name "figs/${action}_py.gif" --print_edge
# python src/forward_kinematics.py --sample_name samples.h5 --action_name "${action}_0"
# convert "figs/${action}_py.gif" "cvtpng/${action}_py.png"
done

# for i in 0 1 2 3 4 5 6 7
# do
#   python src/forward_kinematics.py --sample_name samples_tf.h5 --action_name "sittingdown_${i}" --save --save_name "figs/${i}_py.gif"
# done

# for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
# do
#   python src/forward_kinematics.py --sample_name samples_val_visual.h5 --action_name "validation_${i}" --save --save_name "figs/${i}_py.gif"
# done
