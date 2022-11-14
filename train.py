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
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from parallelformers import parallelize

# My pre-defined functions
from model import ByT5Reggressor
from utils import *
from dataset import train_valid_test_split, create_dataloaders

def train(model, optimizer, scheduler, loss_function, 
          epochs, train_dataloader, valid_dataloader, device, clip_value=2):
    
    training_starting_time = time.time()
    training_stats = []
    validation_predictions = {}

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
            loss = loss_function(predictions.squeeze(), batch_labels.squeeze())
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

        # Validation
        print("")
        print("Running Validation")

        valid_start_time = time.time()

        model.eval()

        total_eval_loss = 0
        eval_steps = 0
        predictions_list = []

        for step, batch in enumerate(valid_dataloader):
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
            with torch.no_grad():
                predictions = model(batch_inputs, batch_masks)
                loss = loss_function(predictions.squeeze(), batch_labels.squeeze()) 
            total_eval_loss += loss.item()
            predictions = predictions.detach().cpu().numpy()
            for i in predictions:
                predictions_list.append(i[0])
            # batch_labels = batch_labels.to("cpu").numpy()
        average_valid_loss = total_eval_loss / len(valid_dataloader)
        valid_ending_time = time.time()
        validation_time = time_format(valid_ending_time-valid_start_time)
        print(f"Average validation loss = {average_valid_loss}")
        print(f"validation took {validation_time}")

        training_stats.append(
            {
                "epoch": epoch + 1,
                "training loss": average_training_loss,
                "validation loss": average_valid_loss,
                "training time": training_time,
                "validation time": validation_time
            }
        )

        validation_predictions.update(
            {
                f"epoch_{epoch+1}": predictions_list
            }
        )
        
        # Save stats per epoch
        if (epoch+1) % 10 == 0:
            saveCSV(pd.DataFrame(data=training_stats), f"statistics/{property_name}/{model_name}/baseline/training_statistics_for_{model_name}_finetuned_{regressor_type}_using_{loss_type}_loss_after_{epoch+1}_epochs.csv")
            saveCSV(pd.DataFrame(validation_predictions), f"statistics/{property_name}/{model_name}/baseline/validation_statistics_for_{model_name}_finetuned_{regressor_type}_using_{loss_type}_loss_after_{epoch+1}_epochs.csv")
        else:
            continue

        torch.save(model.state_dict(), f"model_checkpoints/{property_name}/{model_name}/baseline/{model_name}_finetuned_{regressor_type}_using_{loss_type}_loss_after_{epoch}_epochs.pt")

    train_ending_time = time.time()
    total_training_time = train_ending_time-training_starting_time

    print("")
    print("Training complete")
    print(f"Finetuning {model_name} took {time_format(total_training_time)}")
    
    return training_stats, validation_predictions

if __name__ == "__main__":
    # Set parameters
    # Specify the task
    property_name = "formation_energy" # Default
    property_value = "formation_energy_per_atom" # Default

    prop_data_dir = f"data/property/{property_name}/{property_value}"
    mat_prop_dir = f"data/property/{property_name}/mat_ids_property"
    mat_descr_dir = f"data/property/{property_name}/mat_ids_description"

    n_classes = 1
    batch_size = 64
    max_length = 512
    
    train_data, valid_data, test_data = train_valid_test_split(
        prop_data_dir=prop_data_dir,
        mat_prop_dir=mat_prop_dir,
        mat_descr_dir=mat_descr_dir,
        split_ratio=[7,2,1]
    )
    
    print(f"train data = {len(train_data)} samples")
    print(f"valid data = {len(valid_data)} samples")

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
    
    # Set the number of epochs
    epochs = 60

    # Define the loss functions: using "mean absolute error:mae" and "mean square error:mse" losses
    loss_type = "mae"
    if loss_type == "mae":
        loss_function = nn.L1Loss()
    elif loss_type == "mse":
        loss_function = nn.MSELoss()

    # Specify the model (byt5-small/byt5-base/byt5-large/byt5-3b/byt5-11b)
    # model_name = "byt5-small" # Default model
    model_names = ["t5-small"] # Default model , "t5-base", "t5-large", "t5-xl", "t5-xxl"

    for model_name in model_names:
        if model_name == "t5-small": #  params
            base_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small")
            tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-small")
            base_model_output_size = 512
        elif model_name == "t5-base": #  params
            base_model = T5EncoderModel.from_pretrained("google/t5-v1_1-base")
            tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base")
            base_model_output_size = 768
        elif model_name == "t5-large": #  params
            base_model = T5EncoderModel.from_pretrained("google/t5-v1_1-large")
            tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-large")
            base_model_output_size = 1024

        # if model_name == "byt5-small": # 300M params
        #     base_model = T5EncoderModel.from_pretrained("google/byt5-small")
        #     tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        #     base_model_output_size = 1472
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

        # Instantiate the model
        model = ByT5Reggressor(base_model, base_model_output_size, n_classes, regressor_type, drop_rate=0.1) # add arguments later and put it in mai
        # parallelize(model, num_gpus=4, fp16=True, verbose='detail')
        model.to(device)

        # Load data
        train_dataloader = create_dataloaders(tokenizer, train_data, max_length, batch_size)
        valid_dataloader = create_dataloaders(tokenizer, valid_data, max_length, batch_size)
        # test_dataloader = create_dataloaders(tokenizer, test_data, max_length, batch_size)

        # Define the optimizer
        optimizer = AdamW(
            model.parameters(),
            lr = 0.001, # will check if it is needed to add "eps=1e-8" after lr
            eps = 1e-8
        )

        # Training steps = 262,144 from ByT5 paper
        # Set up the scheduler
        total_training_steps = len(train_dataloader) * epochs # Define train_dataloader later
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_training_steps 
        )

        training_stats, validation_predictions = train(model, optimizer, scheduler, loss_function, 
            epochs, train_dataloader, valid_dataloader, device, clip_value=2)

        # print the model parameters
        model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(model_trainable_params)

        df_traing_stats = pd.DataFrame(data=training_stats)

        # df_traing_stats = df_traing_stats.set_index('epoch')

        df_predictions_stats = pd.DataFrame(validation_predictions)

        # print(df_traing_stats )
        # print(df_predictions_stats)

        # Save stats
        saveCSV(df_traing_stats, f"statistics/{property_name}/{model_name}/baseline/training_statistics_for_{model_name}_finetuned_{regressor_type}_using_{loss_type}_loss_with_{epochs}_epochs.csv")
        saveCSV(df_predictions_stats, f"statistics/{property_name}/{model_name}/baseline/validation_statistics_for_{model_name}_finetuned_{regressor_type}_using_{loss_type}_loss_with_{epochs}_epochs.csv")


        # Save the trained model for inference
        torch.save(model.state_dict(), f"model_checkpoints/{property_name}/{model_name}/baseline/{model_name}_finetuned_{regressor_type}_using_{loss_type}_loss_with_{epochs}_epochs.pt")


