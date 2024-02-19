#!/usr/bin/env bash

TRAIN_PATH="data/samples/textedge_prop_mp22_train.csv" # few examples just for testing purporse only, not intended to reproduce the results
TEST_PATH="data/samples/textedge_prop_mp22_test.csv" # few examples just for testing purporse only, not intended to reproduce the results
TASK_NAME="regression" # the task name can also be set to "classification"
PROPERTY="band_gap" # the property can also be set to "volume" or "is_gap_direct".
CKPT_PATH="checkpoints/samples/$TASK_NAME/best_checkpoint_for_$PROPERTY.tar.gz" # path to the best model 

python llmprop_evaluate.py \
--train_data_path $TRAIN_PATH \
--test_data_path $TEST_PATH \
--task_name $TASK_NAME \
--property $PROPERTY \
--checkpoint $CKPT_PATH