"""
A function to prepare the dataloaders
"""
# Import packages
import glob
import torch
# import spacy
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import *

np.random.seed(42)

def tokenize(tokenizer, dataframe, max_length, pooling='cls'):
    """
    1. Takes in the the list of input sequences and return 
    the input_ids and attention masks of the tokenized sequences
    2. max_length = the max length of each input sequence 
    """
    if pooling == 'cls':
        encoded_corpus = tokenizer(text=["[CLS] " + str(descr) for descr in dataframe.description.tolist()],
                                    add_special_tokens=True,
                                    padding='max_length',
                                    truncation='longest_first',
                                    max_length=max_length, # According to ByT5 paper
                                    return_attention_mask=True)
    elif pooling == 'mean':
        encoded_corpus = tokenizer(text=dataframe.description.tolist(),
                                    add_special_tokens=True,
                                    padding='max_length',
                                    truncation='longest_first',
                                    max_length=max_length, # According to ByT5 paper
                                    return_attention_mask=True) 
    input_ids = encoded_corpus['input_ids']
    attention_masks = encoded_corpus['attention_mask']

    return input_ids, attention_masks

def create_dataloaders(tokenizer, dataframe, max_length, batch_size, property_value="band_gap", pooling='cls', normalize=False, normalizer='z_norm'):
    """
    Dataloader which arrange the input sequences, attention masks, and labels in batches
    and transform the to tensors
    """
    input_ids, attention_masks = tokenize(tokenizer, dataframe, max_length, pooling=pooling)
    labels = dataframe[property_value].to_numpy()

    input_tensor = torch.tensor(input_ids)
    mask_tensor = torch.tensor(attention_masks)
    labels_tensor = torch.tensor(labels)

    if normalize:
        if normalizer == 'z_norm':
            normalized_labels = z_normalizer(labels_tensor)
        elif normalizer == 'mm_norm':
           normalized_labels = min_max_scaling(labels_tensor)
        elif normalizer == 'ls_norm':
            normalized_labels = log_scaling(labels_tensor)
        elif normalizer == 'no_norm':
            normalized_labels = labels_tensor

        dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor, normalized_labels)
    else:
        dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # Set the shuffle to False for now since the labels are continues values check later if this may affect the result

    return dataloader
