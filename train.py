"""
Set up the training code 
"""
import time
import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm

import evaluate # A Huggingface module that contains most the evaluation metrics (pip install evaluate)
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, T5EncoderModel

# My pre-defined functions
from model import ByT5Reggressor
from utils import *
from dataset import create_dataloaders

def train(model, optimizer, scheduler, loss_function, 
          epochs, train_dataloader, device, clip_value=2):
    
    training_starting_time = time.time()

    for epoch in range(epochs):
        print(f"========== Epoch {epoch+1}/{epochs} =========")
        print("Training...")

        epoch_starting_time = time.time() 

        best_loss = 1e10 # Set the best loss variable which record the best loss for each epoch
        total_training_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            print(f"Step {step+1}/{len(train_dataloader)}")

            # Progress update every 20 batches
            if step % 20 == 0 and not step == 0:
                # Calculate elapsed time in minutes
                ending_time = time.time()
                elapsed_time = time_format(ending_time-epoch_starting_time)

                # Check progress
                print(f"Batch {step}/{len(train_dataloader)}. Elapsed time:{elapsed_time}")
            
            
            model.zero_grad() # Resetting the gradients of the previous step
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
            predictions = model(batch_inputs, batch_masks)
            loss = loss_function(predictions, batch_labels)
            # print("outputs size = ", predictions.size())
            # print("outputs size squeezed = ", predictions.squeeze().size())
            # print(predictions.squeeze())
            # print("labels size squeezed = ", batch_labels.squeeze().size())
            # print(batch_labels)
            # print(batch_labels.squeeze())
            # for refs, preds in zip(batch_labels.squeeze(), predictions.squeeze()):
            #     loss_function.add(references=refs, predictions=preds)
            # loss = loss_function.compute()
            total_training_loss += loss.item()
            loss.backward()
            # clip_grad_norm(model.parameters(), clip_value) # Preventing vanishing/exploding gradient issues
            optimizer.step()
            scheduler.step()
        
        average_training_loss = total_training_loss/len(train_dataloader)
        epoch_ending_time = time.time()
        training_time = time_format(epoch_ending_time - epoch_starting_time)

        print(f"Average training loss = {average_training_loss}")
        print(f"Training for this epoch took {training_time}")

    train_ending_time = time.time()
    print(f"Finetuning {model_name} took {train_ending_time-training_starting_time}")
    # return model

if __name__ == "__main__":
    # Set parameters
    n_classes = 1
    batch_size = 32
    max_length = 1024
    train_data = pd.DataFrame({"description":["Cs is Tungsten structured and crystallizes in the cubic Im̅3m space group. Cs is bonded in a body-centered cubic geometry to eight equivalent Cs atoms. All Cs–Cs bond lengths are 5.29 Å.",
    "Pd is Copper structured and crystallizes in the cubic Fm̅3m space group. Pd is bonded to twelve equivalent Pd atoms to form a mixture of edge, face, and corner-sharing PdPd₁₂ cuboctahedra. All Pd–Pd bond lengths are 2.80 Å.",
    "Pd is Copper structured and crystallizes in the cubic Fm̅3m space group. Pd is bonded to twelve equivalent Pd atoms to form a mixture of edge, face, and corner-sharing PdPd₁₂ cuboctahedra. All Pd–Pd bond lengths are 2.80 Å."],
    "value":[0.038769612068965564, 0.0, 0.0]})
    # valid_data = pd.read_csv()
    # test_data = pd.read_csv()

    # Specify the task
    property_name = "formation_energy" # Default 

    
    # Specify the model (byt5-small/byt5-base/byt5-large/byt5-3b/byt5-11b)
    model_name = "byt5-small" # Default model

    if model_name == "byt5-small": # 300M params
        base_model = T5EncoderModel.from_pretrained("google/byt5-small")
        tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        base_model_output_size = 1472
    # elif model_name == "byt5-base": #580 params
    #     base_model = T5EncoderModel.from_pretrained("google/byt5-base")
    #     tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
    #     base_model_output_size = 1536
    # elif model_name == "byt5-large": # 1.2B params
    #     base_model = T5EncoderModel.from_pretrained("google/byt5-large")
    #     tokenizer = AutoTokenizer.from_pretrained("google/byt5-large")
    #     base_model_output_size = 1536
    # elif model_name == "byt5-xl": # 3.7B params
    #     base_model = T5EncoderModel.from_pretrained("google/byt5-xl")
    #     tokenizer = AutoTokenizer.from_pretrained("google/byt5-xl")
    #     base_model_output_size = 2560
    # elif model_name == "byt5-xxl": # 13B params
    #     base_model = T5EncoderModel.from_pretrained("google/byt5-xxl")
    #     tokenizer = AutoTokenizer.from_pretrained("google/byt5-xxl")
    #     base_model_output_size = 4672
    
    # Load data
    train_dataloader = create_dataloaders(tokenizer, train_data, max_length, batch_size)
    # valid_dataloader = create_dataloaders(tokenizer, valid_data, max_length, batch_size)
    # test_dataloader = create_dataloaders(tokenizer, test_data, max_length, batch_size)

    # Build a regression layer (linear/MLP) over ByT5 Encoder
    regressor_type = "linear" # Default

    # Check if the GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Number of available devices: {torch.cuda.device_count()}')
        print(f'Current device is: {torch.cuda.current_device()}')
    else:
        print("No GPU available, please connect to the GPU first or continue to use CPU instead")
        device = torch.device("cpu")

    # Instantiate the model
    model = ByT5Reggressor(base_model, base_model_output_size, n_classes, regressor_type, drop_rate=0.1) # add arguments later and put it in mai
    model.to(device)

    # Define the optimizer
    optimizer = AdamW(
        model.parameters(),
        lr = 0.001, # will check if it is needed to add "eps=1e-8" after lr
        eps = 1e-8
    )
    
    # Set the number of epochs
    epochs = 2

    # Training steps = 262,144 from ByT5 paper
    # Set up the scheduler
    total_training_steps = len(train_dataloader) * epochs # Define train_dataloader later
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_training_steps 
    )

    # Define the loss functions: using "mean absolute error:mae" and "mean square error:mse" losses
    loss_type = "mae"
    if loss_type == "mae":
        loss_function = nn.L1loss()
    elif loss_type == "mse":
        loss_function = nn.MSELoss()

    train(model, optimizer, scheduler, loss_function, 
          epochs, train_dataloader, device, clip_value=2)

