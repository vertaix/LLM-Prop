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

import matplotlib.pyplot as plt

import evaluate # A Huggingface module that contains most the evaluation metrics (pip install evaluate)
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

# My pre-defined functions
from model import T5Reggressor
from utils import *
from dataset import train_valid_test_split, create_dataloaders

def train(model, optimizer, scheduler, loss_function, 
          epochs, train_dataloader, valid_dataloader, device, clip_value=2):
    
    training_starting_time = time.time()
    training_stats = []
    validation_predictions = {}
    
    best_loss = 1e10 # Set the best loss variable which record the best loss for each epoch

    for epoch in range(epochs):
        print(f"========== Epoch {epoch+1}/{epochs} =========")
        print("Training...")

        epoch_starting_time = time.time() 

        total_training_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            print(f"Step {step+1}/{len(train_dataloader)}")

            # # Progress update every 20 batches
            # if step % 20 == 0 and not step == 0:
            #     # Calculate elapsed time in minutes
            #     ending_time = time.time()
            #     elapsed_time = time_format(ending_time-epoch_starting_time)

            #     # Check progress
            #     print(f"Batch {step}/{len(train_dataloader)}. Elapsed time:{elapsed_time}")
            
            model.zero_grad() # Resetting the gradients of the previous step
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
            # print(batch_inputs.size(), batch_masks.size()) (4,1)->(4,)
            predictions = model(batch_inputs, batch_masks)
            loss = loss_function(predictions.squeeze(), batch_labels.squeeze())
            total_training_loss += loss.item()
            loss.backward()
            if regressor_type == "gru":
                clip_grad_norm(model.parameters(), clip_value) # Preventing vanishing/exploding gradient issues
            optimizer.step()
            scheduler.step()
        
        average_training_loss = total_training_loss/len(train_dataloader)
        epoch_ending_time = time.time()
        training_time = time_format(epoch_ending_time - epoch_starting_time)

        print(f"Average training loss = {average_training_loss}")
        print(f"Training for this epoch took {training_time}")

        # Validation
        print("")
        print("Running Validation ....")

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
        
        if average_valid_loss <= best_loss:
            best_loss = average_valid_loss
            best_epoch = epoch+1
        else:
            best_loss = best_loss

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

        # Save the trained model for inference 
        torch.save(model.state_dict(), f"model_checkpoints/main_paper/{property_name}/{model_name}/{regressor_type}/{model_name}_finetuned_{regressor_type}_using_{loss_type}_loss_after_{epoch}_epochs.pt")

        # Save stats per epoch
        if (epoch+1) % 10 == 0:
            saveCSV(pd.DataFrame(data=training_stats), f"statistics/main_paper/{property_name}/{model_name}/{regressor_type}/training_statistics_for_{model_name}_finetuned_{regressor_type}_using_{loss_type}_loss_after_{epoch+1}_epochs.csv")
            saveCSV(pd.DataFrame(validation_predictions), f"statistics/main_paper/{property_name}/{model_name}/{regressor_type}/validation_statistics_for_{model_name}_finetuned_{regressor_type}_using_{loss_type}_loss_after_{epoch+1}_epochs.csv")
        else:
            continue

        

    train_ending_time = time.time()
    total_training_time = train_ending_time-training_starting_time

    print("")
    print("Training complete")
    print(f"Finetuning {model_name} took {time_format(total_training_time)}")
    print(f"The lowest valid loss achieved is {best_loss} at {best_epoch} epoch")
    
    return training_stats, validation_predictions

if __name__ == "__main__":
    # Set parameters
    # Specify the task
    property_name = "formation_energy" # Default
    property_value = "formation_energy_per_atom" # Default

    prop_data_dir = f"data/property/{property_name}/{property_value}"
    mat_prop_dir = f"data/property/{property_name}/mat_ids_property"
    mat_descr_dir = f"data/property/{property_name}/mat_ids_description"

    batch_size = 64
    max_length = 512
    hidden_dim = 128
    n_layers = 2
    n_filters = 100
    filter_sizes = [3,4,5]
    
    train_data, valid_data, test_data = train_valid_test_split(
        prop_data_dir=prop_data_dir,
        mat_prop_dir=mat_prop_dir,
        mat_descr_dir=mat_descr_dir,
        split_ratio=[8,1,1]
    )

    # train_data =  pd.read_csv(f"data/property/{property_name}/{property_value}/normalized_train.csv")
    # valid_data = pd.read_csv(f"data/property/{property_name}/{property_value}/normalized_valid.csv")

    print(f"train data = {len(train_data)} samples")
    print(f"valid data = {len(valid_data)} samples")

    # regressor types
    regressor_types = ["linear", "gru", "cnn", "mlp"] # Default
    # regressor_type = "rnn"

    # Check if the GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Number of available devices: {torch.cuda.device_count()}')
        print(f'Current device is: {torch.cuda.current_device()}')
    else:
        print("No GPU available, please connect to the GPU first or continue to use CPU instead")
        device = torch.device("cpu")
    
    # Set the number of epochs
    epochs = 100

    # Define the loss functions: using "mean absolute error:mae" and "mean square error:mse" losses
    loss_type = "mae"
    if loss_type == "mae":
        loss_function = nn.L1Loss()
    elif loss_type == "mse":
        loss_function = nn.MSELoss()

    # Specify the model (t5-small/t5-base/t5-large/t5-xl/t5-xxl)
    # model_name = "t5-small" # Default model
    model_names = ["t5-small"] # Default model , "t5-base", "t5-large", "t5-xl", "t5-xxl"
    freeze = False # a boolean variable to determine if we freeze the pre-trained T5 parameters and only train the regressor parameters

    modified_tokenizer = AutoTokenizer.from_pretrained("tokenizers/new_pretrained_t5_tokenizer_on_modified_c4_and_mat_descriptions_52k_vocab")

    # split the train data into 20%, 40%, 60%, and 80%
    # train_data_list = [
    #     train_data.iloc[:int(0.2*len(train_data))],
    #     train_data.iloc[:int(0.4*len(train_data))], 
    #     train_data.iloc[:int(0.6*len(train_data))],
    #     train_data.iloc[:int(0.8*len(train_data))],
    # ]
    # for train_data in train_data_list:

    for regressor_type in regressor_types:
        if regressor_type=="linear":
            learning_rate = 0.001
            drop_rate = 0.1
        elif regressor_type=="gru":
            learning_rate = 1e-4
            drop_rate = 0.5
        elif regressor_type=="cnn":
            learning_rate = 1e-3
            drop_rate = 0.2
        elif regressor_type=="mlp":
            learning_rate = 1e-3
            drop_rate = 0.5

        for model_name in model_names:
            if model_name == "t5-small": #  params
                base_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small")
                base_model_output_size = 512
            elif model_name == "t5-base": #  params
                base_model = T5EncoderModel.from_pretrained("google/t5-v1_1-base")
                base_model_output_size = 768
            elif model_name == "t5-large": #  params
                base_model = T5EncoderModel.from_pretrained("google/t5-v1_1-large")
                base_model_output_size = 1024

            # freeze the pre-trained LM's parameters
            if freeze:
                for param in base_model.parameters():
                    param.requires_grad = False

            # Resizing the model input embeddings matrix to adapt to newly added tokens by the new tokenizer
            # This is to avoid the "RuntimeError: CUDA error: device-side assert triggered" error
            base_model.resize_token_embeddings(len(modified_tokenizer))

            # Instantiate the model
            model = T5Reggressor(base_model, base_model_output_size, regressor_type, hidden_dim, filter_sizes, n_layers, n_filters, drop_rate=drop_rate) # add arguments later and put it in mai
            
            model.to(device)

            # Load data
            train_dataloader = create_dataloaders(modified_tokenizer, train_data, max_length, batch_size)
            valid_dataloader = create_dataloaders(modified_tokenizer, valid_data, max_length, batch_size)
            # test_dataloader = create_dataloaders(tokenizer, test_data, max_length, batch_size)

            # Define the optimizer
            optimizer = AdamW(
                model.parameters(),
                lr = learning_rate, # will check if it is needed to add "eps=1e-8" after lr
                eps = 1e-8
            )

            # Set up the scheduler
            total_training_steps = len(train_dataloader) * epochs 
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_training_steps 
            )

            training_stats, validation_predictions = train(model, optimizer, scheduler, loss_function, 
                epochs, train_dataloader, valid_dataloader, device, clip_value=2)

            # print the model parameters
            model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters = {model_trainable_params}")

            df_traing_stats = pd.DataFrame(data=training_stats)

            df_predictions_stats = pd.DataFrame(validation_predictions)

            # print(df_traing_stats )
            # print(df_predictions_stats)

            # save the stats figure
            plt.plot(df_traing_stats["epoch"], df_traing_stats["training loss"], label="Traning Loss")
            plt.plot(df_traing_stats["epoch"], df_traing_stats["validation loss"], label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("MAE Loss")
            plt.legend()
            plt.savefig(f"figures/main_paper/{property_name}/{model_name}/{regressor_type}/training_stats.png", dpi=300)
            plt.show()

            print("-"*100)

            # # Save stats
            # saveCSV(df_traing_stats, f"statistics/{property_name}/{model_name}/ablation/training_statistics_for_{model_name}_finetuned_{regressor_type}_using_{loss_type}_loss_with_{epochs}_epochs_on_{len(train_data)}_examples.csv")
            # saveCSV(df_predictions_stats, f"statistics/{property_name}/{model_name}/ablation/validation_statistics_for_{model_name}_finetuned_{regressor_type}_using_{loss_type}_loss_with_{epochs}_epochs_on_{len(train_data)}_examples.csv")

            # # Save the trained model for inference
            # torch.save(model.state_dict(), f"model_checkpoints/{property_name}/{model_name}/ablation/{model_name}_finetuned_{regressor_type}_using_{loss_type}_loss_with_{epochs}_epochs_on_{len(train_data)}_examples.csv.pt")

