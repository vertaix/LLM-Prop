"""
A function to prepare the dataloaders
"""
# Import packages
import torch
from torch.utils.data import DataLoader, TensorDataset

def tokenize(tokenizer, dataframe, max_length):
    """
    1. Takes in the the list of input sequences and return 
    the input_ids and attention masks of the tokenized sequences
    2. max_length = the max length of each input sequence 
    (1024 for char-based tokenizer (default for ByT5) and 512 for token-based tokenizer)
    """
    encoded_corpus = tokenizer(text=dataframe.description.tolist(),
                                # add_special_tokens=True,
                                padding='max_length',
                                truncation='longest_first',
                                max_length=max_length, # According to ByT5 paper
                                return_attention_mask=True)
    input_ids = encoded_corpus['input_ids']
    attention_masks = encoded_corpus['attention_mask']

    return input_ids, attention_masks

def create_dataloaders(tokenizer, dataframe, max_length, batch_size):
    """
    Dataloader which arrange the input sequences, attention masks, and labels in batches
    and transform the to tensors
    """
    input_ids, attention_masks = tokenize(tokenizer, dataframe, max_length)
    labels = dataframe.value.to_numpy()

    input_tensor = torch.tensor(input_ids)
    mask_tensor = torch.tensor(attention_masks)
    labels_tensor = torch.tensor(labels)

    dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # Set the shuffle to False for now since the labes are continues values check later if this may affect the result

    return dataloader
