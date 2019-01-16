#!/bin/bash

#Train the model
python src/translate.py --action walking --seq_length_out 25 --iterations 10000 --residual_velocities
