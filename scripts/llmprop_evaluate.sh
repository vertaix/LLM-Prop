#!/usr/bin/env bash

TRAIN_PATH="data/samples/textedge_prop_mp22_train.csv"
TEST_PATH="data/samples/textedge_prop_mp22_test.csv"
TASK_NAME="regression" # the task name can also be set to "classification"
PROPERTY="band_gap" # the property can also be set to "volume" or "is_gap_direct". Note that if the task name is set to classification, only "is_gap_direct" is allowed here. And if the task name is set to regression, only "band_gap" or "volume" is allowed here.
CKPT_PATH="checkpoints/samples/$TASK_NAME/best_checkpoint_for_$PROPERTY.tar.gz" # path to the best model if the property to be predicted

python llmprop_evaluate.py \
--train_data_path $TRAIN_PATH \
--test_data_path $TEST_PATH \
--task_name $TASK_NAME \
--property $PROPERTY \
--checkpoint $CKPT_PATH