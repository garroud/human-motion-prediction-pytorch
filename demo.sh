#!/bin/bash

#Train the model

NOW=$(date +"%Y%m%d%H%M%S")
python src/translate.py --learning_rate 0.005 --residual_velocities 2>&1 | tee logs/log-$NOW.txt
