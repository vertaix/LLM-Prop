#!/usr/bin/env bash

TRAIN_PATH="data/samples/textedge_prop_mp22_train.csv"
VALID_PATH="data/samples/textedge_prop_mp22_valid.csv"
TEST_PATH="data/samples/textedge_prop_mp22_test.csv"
EPOCHS=2 # the default epochs is 200 to get the best performance
PROPERTY="band_gap" # specify the property name

python llmprop_train.py \
--train_data_path $TRAIN_PATH \
--valid_data_path $VALID_PATH \
--test_data_path $TEST_PATH \
--epochs $EPOCHS \
--property $PROPERTY