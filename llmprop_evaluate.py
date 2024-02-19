"""
Codes to evaluate the trained models on held out test set
"""
# Import necessary packages
import re
import time
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from statistics import stdev

# pre-defined functions
from llmprop_utils import *
from llmprop_dataset import *
from llmprop_args_parser import *
from llmprop_train import evaluate
from llmprop_model import T5Predictor

if __name__ == "__main__":
    print("======= Evaluating on test set ========")

    # check if the GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Number of available devices: {torch.cuda.device_count()}')
        print(f'Current device is: {torch.cuda.current_device()}')
        print("Testing on", torch.cuda.device_count(), "GPUs!")
        print('-'*50)
    else:
        print("No GPU available, please connect to the GPU first or continue to use CPU instead")
        print('-'*50)
        device = torch.device("cpu")

    # parse Arguments
    args = args_parser()
    config = vars(args)

    # set parameters
    batch_size = config.get('bs')
    max_length = config.get('max_len')
    drop_rate = config.get('dr')
    preprocessing_strategy = config.get('preprocessing_strategy')
    tokenizer_name = config.get('tokenizer')
    pooling = config.get('pooling')
    normalizer_type = config.get('normalizer')
    property = config.get('property_name')
    task_name = config.get('task_name')
    train_data_path = config.get('train_data_path')
    test_data_path = config.get('test_data_path')
    best_model_path = config.get('checkpoint')

    # prepare the data
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    # check property type to determine the task name (whether it is regression or classification)
    if test_data[property].dtype == 'bool':
        task_name = 'classification'

        #converting True->1.0 and False->0.0
        train_data[property] = train_data[property].astype(float)
        test_data[property] = test_data[property].astype(float)
    else:
        task_name = 'regression'

    train_labels_array = np.array(train_data[property])
    train_labels_mean = torch.mean(torch.tensor(train_labels_array))
    train_labels_std = torch.std(torch.tensor(train_labels_array))
    train_labels_min = torch.min(torch.tensor(train_labels_array))
    train_labels_max = torch.max(torch.tensor(train_labels_array))

    if preprocessing_strategy == "none":
        test_data = test_data
        print(test_data['description'][0])
        print('-'*50)

    elif preprocessing_strategy == "bond_lengths_replaced_with_num":
        test_data['description'] = test_data['description'].apply(replace_bond_lengths_with_num)
        print(test_data['description'][0])
        print('-'*50)

    elif preprocessing_strategy == "bond_angles_replaced_with_ang":
        test_data['description'] = test_data['description'].apply(replace_bond_angles_with_ang) 
        print(test_data['description'][0])
        print('-'*50)

    elif preprocessing_strategy == "no_stopwords":
        stopwords = get_cleaned_stopwords()
        test_data['description'] = test_data['description'].apply(remove_mat_stopwords)
        print(test_data['description'][0])
        print('-'*50)

    elif preprocessing_strategy == "no_stopwords_and_lengths_and_angles_replaced":
        stopwords = get_cleaned_stopwords()
        test_data['description'] = test_data['description'].apply(remove_mat_stopwords)
        test_data['description'] = test_data['description'].apply(replace_bond_lengths_with_num)
        test_data['description'] = test_data['description'].apply(replace_bond_angles_with_ang)
        print(test_data['description'][0])
        print('-'*50) 

    # define loss functions
    mae_loss_function = nn.L1Loss()

    freeze = False # a boolean variable to determine if we freeze the pre-trained T5 weights

    # define the tokenizer
    if tokenizer_name == 't5_tokenizer': 
        tokenizer = AutoTokenizer.from_pretrained("t5-small")

    elif tokenizer_name == 'modified':
        tokenizer = AutoTokenizer.from_pretrained("tokenizers/t5_tokenizer_trained_on_modified_part_of_C4_and_textedge")

    # add defined special tokens to the tokenizer
    if pooling == 'cls':
        tokenizer.add_tokens(["[CLS]"])

    if preprocessing_strategy == "bond_lengths_replaced_with_num":
        tokenizer.add_tokens(["[NUM]"]) # special token to replace bond lengths

    elif preprocessing_strategy == "bond_angles_replaced_with_ang":
        tokenizer.add_tokens(["[ANG]"]) # special token to replace bond angles

    elif preprocessing_strategy == "no_stopwords_and_lengths_and_angles_replaced":
        tokenizer.add_tokens(["[NUM]"])
        tokenizer.add_tokens(["[ANG]"]) 

    print(f"test data = {len(test_data)} samples")
    print('-'*50)
    print(f"testing on {get_sequence_len_stats(test_data, tokenizer, max_length)}% samples with whole sequence")
    print('-'*50)

    print("labels statistics on training set:")
    print("Mean:", train_labels_mean)
    print("Standard deviation:", train_labels_std)
    print("Max:", train_labels_max)
    print("Min:", train_labels_min)
    print("-"*50)

    # define the model
    base_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small")
    base_model_output_size = 512

    # freeze the pre-trained LM's parameters
    if freeze:
        for param in base_model.parameters():
            param.requires_grad = False

    # resizing the model input embeddings matrix to adapt to newly added tokens by the new tokenizer
    # this is to avoid the "RuntimeError: CUDA error: device-side assert triggered" error
    base_model.resize_token_embeddings(len(tokenizer))

    # loading the checkpoint of the pretrained model
    if "tar.gz" in best_model_path:
        decompressTarCheckpoints(best_model_path)
        best_model_path = best_model_path[0:-7] + ".pt"

    best_model = T5Predictor(base_model, base_model_output_size, drop_rate=drop_rate, pooling=pooling)

    device_ids = [d for d in range(torch.cuda.device_count())]

    if torch.cuda.is_available():
        best_model = nn.DataParallel(best_model, device_ids=device_ids).cuda()

    if isinstance(best_model, nn.DataParallel):
        best_model.module.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)), strict=False)
    else:
        best_model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)), strict=False) 
        best_model.to(device)

    # create test set dataloaders
    test_dataloader = create_dataloaders(
        tokenizer, 
        test_data, 
        max_length, 
        batch_size, 
        property_value=property, 
        pooling=pooling
    )

    _, test_performance = evaluate(best_model, mae_loss_function, test_dataloader, train_labels_mean, train_labels_std, train_labels_min, train_labels_max, property, device, task_name, normalizer=normalizer_type)
    