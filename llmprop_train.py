"""
Set up the training code 
"""
import re
import glob
import time
import datetime
from datetime import timedelta

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import SGD

import matplotlib.pyplot as plt

# add the progress bar
from tqdm import tqdm

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
from tokenizers.pre_tokenizers import Whitespace

pre_tokenizer = Whitespace()

# pre-defined functions
from llmprop_model import T5Predictor
from llmprop_utils import *
from llmprop_dataset import *
from llmprop_args_parser import *

# for metrics
from torchmetrics.classification import BinaryAUROC

# set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def train(
    model, 
    optimizer, 
    scheduler, 
    bce_loss_function, 
    mae_loss_function,
    epochs, 
    train_dataloader, 
    valid_dataloader, 
    device,  
    normalizer="z_norm"
):
    
    training_starting_time = time.time()
    training_stats = []
    validation_predictions = {}
    
    best_loss = 1e10 # Set the best loss variable which record the best loss for each epoch
    best_roc = 0.0

    for epoch in range(epochs):
        print(f"========== Epoch {epoch+1}/{epochs} =========")
        print("Training...")

        epoch_starting_time = time.time() 

        total_training_loss = 0
        total_training_mae_loss = 0
        total_training_normalized_mae_loss = 0

        model.train()

        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            
            print(f"Step {step+1}/{len(train_dataloader)}")

            batch_inputs, batch_masks, batch_labels, batch_norm_labels = tuple(b.to(device) for b in batch)

            _, predictions = model(batch_inputs, batch_masks)

            if task_name == 'classification':
                loss = bce_loss_function(predictions.squeeze(), batch_labels.squeeze())
            
            elif task_name == 'regression':
                loss = mae_loss_function(predictions.squeeze(), batch_norm_labels.squeeze())
                
                if normalizer == 'z_norm':
                    predictions_denorm = z_denormalize(predictions, train_labels_mean, train_labels_std)

                elif normalizer == 'mm_norm':
                    predictions_denorm = mm_denormalize(predictions, train_labels_min, train_labels_max)

                elif normalizer == 'ls_norm':
                    predictions_denorm = ls_denormalize(predictions)

                elif normalizer == 'no_norm':
                    loss = mae_loss_function(predictions.squeeze(), batch_labels.squeeze())
                    predictions_denorm = predictions

                mae_loss = mae_loss_function(predictions_denorm.squeeze(), batch_labels.squeeze()) 

            # total training loss on actual output
            if task_name == "classification":
                total_training_loss += loss.item()
            
            elif task_name == "regression":
                total_training_loss += mae_loss.item()

            # back propagate
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # average training loss on actual output
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

        total_eval_mae_loss = 0
        predictions_list = []
        targets_list = []

        for step, batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)

            with torch.no_grad():
                _, predictions = model(batch_inputs, batch_masks)

                if task_name == "classification":
                    predictions_denorm = predictions

                elif task_name == "regression":
                    if normalizer == 'z_norm':
                        predictions_denorm = z_denormalize(predictions, train_labels_mean, train_labels_std)

                    elif normalizer == 'mm_norm':
                        predictions_denorm = mm_denormalize(predictions, train_labels_min, train_labels_max)

                    elif normalizer == 'ls_norm':
                        predictions_denorm = ls_denormalize(predictions)

                    elif normalizer == 'no_norm':
                        predictions_denorm = predictions

            predictions = predictions_denorm.detach().cpu().numpy()
            targets = batch_labels.detach().cpu().numpy()

            for i in range(len(predictions)):
                predictions_list.append(predictions[i][0])
                targets_list.append(targets[i])
        
        valid_ending_time = time.time()
        validation_time = time_format(valid_ending_time-valid_start_time)

        # save model checkpoint and the statistics of the epoch where the model performs the best
        if task_name == "classification":
            valid_performance = get_roc_score(predictions_list, targets_list)
            
            if valid_performance >= best_roc:
                best_roc = valid_performance
                best_epoch = epoch+1

                # save the best model checkpoint
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), f"checkpoints/samples/{task_name}/best_checkpoint_for_{property}.pt")
                else:
                    torch.save(model.state_dict(), f"checkpoints/samples/{task_name}/best_checkpoint_for_{property}.pt")
                
                # save statistics of the best model
                training_stats.append(
                    {
                        "best_epoch": epoch + 1,
                        "training_loss": average_training_loss,
                        "validation_roc_score": valid_performance,
                        "training time": training_time,
                        "validation time": validation_time
                    }
                )

                validation_predictions.update(
                    {
                        f"epoch_{epoch+1}": predictions_list
                    }
                )

                saveCSV(pd.DataFrame(data=training_stats), f"statistics/samples/{task_name}/training_stats_for_{property}.csv")
                saveCSV(pd.DataFrame(validation_predictions), f"statistics/samples/{task_name}/validation_stats_for_{property}.csv")

            else:
                best_roc = best_roc

            print(f"Validation roc score = {valid_performance}")

        elif task_name == "regression":
            predictions_tensor = torch.tensor(predictions_list)
            targets_tensor = torch.tensor(targets_list)
            valid_performance = mae_loss_function(predictions_tensor.squeeze(), targets_tensor.squeeze())
        
            if valid_performance <= best_loss:
                best_loss = valid_performance
                best_epoch = epoch+1

                # save the best model checkpoint
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), f"checkpoints/samples/{task_name}/best_checkpoint_for_{property}.pt")
                else:
                    torch.save(model.state_dict(), f"checkpoints/samples/{task_name}/best_checkpoint_for_{property}.pt")
                
                # save statistics of the best model
                training_stats.append(
                    {
                        "best_epoch": epoch + 1,
                        "training mae loss": average_training_loss,
                        "validation mae loss": valid_performance,
                        "training time": training_time,
                        "validation time": validation_time
                    }
                )

                validation_predictions.update(
                    {
                        f"epoch_{epoch+1}": predictions_list
                    }
                )

                saveCSV(pd.DataFrame(data=training_stats), f"statistics/samples/{task_name}/training_stats_for_{property}.csv")
                saveCSV(pd.DataFrame(validation_predictions), f"statistics/samples/{task_name}/validation_stats_for_{property}.csv")

            else:
                best_loss = best_loss
            
            print(f"Validation mae error = {valid_performance}")
        print(f"validation took {validation_time}")
    train_ending_time = time.time()
    total_training_time = train_ending_time-training_starting_time

    print("")
    print("Training complete")
    print(f"Training LLM_Prop on {task_name} prediction took {time_format(total_training_time)}")

    if task_name == "classification":
        print(f"The lowest roc score achieved on validation set on {property} is {best_roc} at {best_epoch}th epoch")

    elif task_name == "regression":
        print(f"The lowest mae error achieved on validation set on predicting {property} is {best_loss} at {best_epoch}th epoch")
    
    return training_stats, validation_predictions

def evaluate(
    model, 
    mae_loss_function, 
    test_dataloader, 
    train_labels_mean, 
    train_labels_std, 
    property,
    normalizer="z_norm"
):
    test_start_time = time.time()

    model.eval()

    total_test_loss = 0
    predictions_list = []
    targets_list = []

    for step, batch in enumerate(test_dataloader):
        batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)

        with torch.no_grad():
            _, predictions = model(batch_inputs, batch_masks)

            if task_name == "classification":
                predictions_denorm = predictions

            elif task_name == "regression":
                if normalizer == 'z_norm':
                    predictions_denorm = z_denormalize(predictions, train_labels_mean, train_labels_std)

                elif normalizer == 'mm_norm':
                    predictions_denorm = mm_denormalize(predictions, train_labels_min, train_labels_max)

                elif normalizer == 'ls_norm':
                    predictions_denorm = ls_denormalize(predictions)

                elif normalizer == 'no_norm':
                    predictions_denorm = predictions

        predictions = predictions_denorm.detach().cpu().numpy()
        targets = batch_labels.detach().cpu().numpy()

        for i in range(len(predictions)):
            predictions_list.append(predictions[i][0])
            targets_list.append(targets[i])
        
    test_predictions = {f"{property}": predictions_list}

    saveCSV(pd.DataFrame(test_predictions), f"statistics/samples/{task_name}/test_stats_for_{property}.csv")
        
    if task_name == "classification":
        test_performance = get_roc_score(predictions_list, targets_list)
        print(f"The roc score achieved on test set for predicting {property} is {test_performance}")

    elif task_name == "regression":
        predictions_tensor = torch.tensor(predictions_list)
        targets_tensor = torch.tensor(targets_list)
        test_performance = mae_loss_function(predictions_tensor.squeeze(), targets_tensor.squeeze())
        print(f"The mae error achieved on test set for predicting {property} is {test_performance}")

    average_test_loss = total_test_loss / len(test_dataloader)
    test_ending_time = time.time()
    testing_time = time_format(test_ending_time-test_start_time)
    print(f"testing took {testing_time}")

    return predictions_list, test_performance

def replace_bond_lengths_with_num(sentence):
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*Å", "[NUM]", sentence) # Regex pattern to match bond lengths and units
    return sentence.strip()

def replace_bond_angles_with_ang(sentence):
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*°", "[ANG]", sentence) # Regex pattern to match angles and units
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*degrees", "[ANG]", sentence) # Regex pattern to match angles and units
    return sentence.strip()

def replace_bond_lengths_and_angles_with_num_and_ang(sentence):
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*Å", "[NUM]", sentence) # Regex pattern to match bond lengths and units
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*°", "[ANG]", sentence) # Regex pattern to match angles and units
    sentence = re.sub(r"\d+(\.\d+)?(?:–\d+(\.\d+)?)?\s*degrees", "[ANG]", sentence) # Regex pattern to match angles and units
    return sentence.strip()

def get_cleaned_stopwords():
    # from https://github.com/igorbrigadir/stopwords
    stopword_files = glob.glob("../stopwords/en/*.txt")

    all_stopwords_list = set()

    for file_path in stopword_files:
        all_stopwords_list |= set(readTEXT_to_LIST(file_path))

    cleaned_list = {wrd.replace("\n", "").strip() for wrd in all_stopwords_list}
    cleaned_list_for_mat = {wrd for wrd in cleaned_list if not wrd.isdigit()}
    
    return cleaned_list_for_mat

def remove_mat_stopwords(sentence, stopwords_list):
    words = sentence.split()
    words_lower = sentence.lower().split()
    sentence = ' '.join([words[i] for i in range(len(words)) if words_lower[i] not in stopwords_list])
    return sentence

def get_sequence_len_stats(df, tokenizer, max_len):
    training_on = sum(1 for sent in df['description'].apply(tokenizer.tokenize) if len(sent) <= max_len)
    return (training_on/len(df))*100

def get_roc_score(predictions, targets):
    roc_fn = BinaryAUROC(threshold=None)
    x = torch.tensor(targets)
    y = torch.tensor(predictions)
    y = torch.round(torch.sigmoid(y))
    roc_score = roc_fn(y, x)
    return roc_score

if __name__ == "__main__":
    # check if the GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Number of available devices: {torch.cuda.device_count()}')
        print(f'Current device is: {torch.cuda.current_device()}')
    else:
        print("No GPU available, please connect to the GPU first or continue to use CPU instead")
        device = torch.device("cpu")

    # parse Arguments
    args = args_parser()
    config = vars(args)

    # set parameters
    batch_size = config.get('bs')
    max_length = config.get('max_len')
    learning_rate = config.get('lr')
    drop_rate = config.get('dr')
    epochs = config.get('epochs')
    warmup_steps = config.get('warmup_steps')
    preprocessing_strategy = config.get('preprocessing_strategy')
    tokenizer_name = config.get('tokenizer')
    pooling = config.get('pooling')
    scheduler_type = config.get('scheduler')
    normalizer_type = config.get('normalizer')
    property = config.get('property_name')
    optimizer_type = config.get('optimizer')
    task_name = config.get('task_name')

    if task_name == "classification":
        if property not in ["is_gap_direct"]:
            raise Exception("When task_name is 'classification' please set the property name to 'is_gap_direct'")
    elif task_name == "regression":
        if property not in ["band_gap", "volume"]:
            raise Exception("When task_name is 'regression' please set the property name to either 'band_gap' or 'volume'")
    else:
        raise Exception("Please set the task_name to either 'regression' or 'classification'")

    if property in ["is_gap_direct"]:
        if task_name not in ["classification"]:
            raise Exception("Please set the task_name to a 'classification'")
    elif property in ["band_gap", "volume"]:
        if task_name not in ["regression"]:
            raise Exception("Please set the task_name to a 'regression'")
    else:
        raise Exception("Please set the task_name to either 'band_gap', 'volume', or 'is_gap_direct'")

    # prepare the data
    train_data = pd.read_csv("data/samples/textedge_prop_mp22_train.csv")
    valid_data = pd.read_csv("data/samples/textedge_prop_mp22_valid.csv")
    test_data = pd.read_csv("data/samples/textedge_prop_mp22_test.csv")

    if property == "is_gap_direct":
        train_data.loc[train_data["is_gap_direct"] == True, "is_gap_direct"] = 1
        train_data.loc[train_data["is_gap_direct"] == False, "is_gap_direct"] = 0
        train_data.is_gap_direct = train_data.is_gap_direct.astype(float)
        valid_data.loc[valid_data["is_gap_direct"] == True, "is_gap_direct"] = 1
        valid_data.loc[valid_data["is_gap_direct"] == False, "is_gap_direct"] = 0
        valid_data.is_gap_direct = valid_data.is_gap_direct.astype(float)
        test_data.loc[test_data["is_gap_direct"] == True, "is_gap_direct"] = 1
        test_data.loc[test_data["is_gap_direct"] == False, "is_gap_direct"] = 0
        test_data.is_gap_direct = test_data.is_gap_direct.astype(float)
    
    train_labels_array = np.array(train_data[property])
    train_labels_mean = torch.mean(torch.tensor(train_labels_array))
    train_labels_std = torch.std(torch.tensor(train_labels_array))
    train_labels_min = torch.min(torch.tensor(train_labels_array))
    train_labels_max = torch.max(torch.tensor(train_labels_array))

    if preprocessing_strategy == "none":
        train_data = train_data
        valid_data = valid_data

    elif preprocessing_strategy == "bond_lengths_replaced_with_num":
        train_data['description'] = train_data['description'].apply(replace_bond_lengths_with_num)
        valid_data['description'] = valid_data['description'].apply(replace_bond_lengths_with_num)
        test_data['description'] = test_data['description'].apply(replace_bond_lengths_with_num)
        print(train_data['description'][0])
        print('-'*50)
        print(train_data['description'][3])

    elif preprocessing_strategy == "bond_angles_replaced_with_ang":
        train_data['description'] = train_data['description'].apply(replace_bond_angles_with_ang)
        valid_data['description'] = valid_data['description'].apply(replace_bond_angles_with_ang)
        test_data['description'] = test_data['description'].apply(replace_bond_angles_with_ang) 
        print(train_data['description'][0])
        print('-'*50)
        print(train_data['description'][3])

    elif preprocessing_strategy == "no_stopwords":
        stopwords = get_cleaned_stopwords()
        train_data['description'] = train_data.apply(lambda row: remove_mat_stopwords(row['description'], stopwords), axis=1)
        valid_data['description'] = valid_data.apply(lambda row: remove_mat_stopwords(row['description'], stopwords), axis=1)
        test_data['description'] = test_data.apply(lambda row: remove_mat_stopwords(row['description'], stopwords), axis=1)
        print(train_data['description'][0])
        print('-'*50)
        print(valid_data['description'][3])

    elif preprocessing_strategy == "no_stopwords_and_lengths_and_angles_replaced":
        stopwords = get_cleaned_stopwords()
        train_data['description'] = train_data.apply(lambda row: remove_mat_stopwords(row['description'], stopwords), axis=1)
        train_data['description'] = train_data['description'].apply(replace_bond_lengths_with_num)
        train_data['description'] = train_data['description'].apply(replace_bond_angles_with_ang)
        valid_data['description'] = valid_data.apply(lambda row: remove_mat_stopwords(row['description'], stopwords), axis=1)
        valid_data['description'] = valid_data['description'].apply(replace_bond_lengths_with_num)
        valid_data['description'] = valid_data['description'].apply(replace_bond_angles_with_ang)
        test_data['description'] = test_data.apply(lambda row: remove_mat_stopwords(row['description'], stopwords), axis=1)
        test_data['description'] = test_data['description'].apply(replace_bond_lengths_with_num)
        test_data['description'] = test_data['description'].apply(replace_bond_angles_with_ang)
        print(train_data['description'][0])
        print('-'*50)
        print(valid_data['description'][3]) 

    # define loss functions
    mae_loss_function = nn.L1Loss()
    bce_loss_function = nn.BCEWithLogitsLoss()

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
    
    print('-'*50)
    print(f"train data = {len(train_data)} samples")
    print(f"valid data = {len(valid_data)} samples")
    print('-'*50)
    print(f"training on {get_sequence_len_stats(train_data, tokenizer, max_length)}% samples with whole sequence")
    print(f"validating on {get_sequence_len_stats(valid_data, tokenizer, max_length)}% samples with whole sequence")
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

    # instantiate the model
    model = T5Predictor(base_model, base_model_output_size, drop_rate=drop_rate, pooling=pooling)

    device_ids = [d for d in range(torch.cuda.device_count())]

    if torch.cuda.device_count() > 1:
        print("Training on", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        model.to(device)

    # print the model parameters
    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters = {model_trainable_params}")

    # create dataloaders
    train_dataloader = create_dataloaders(
        tokenizer, 
        train_data, 
        max_length, 
        batch_size, 
        property_value=property, 
        pooling=pooling, 
        normalize=True, 
        normalizer=normalizer_type
    )

    valid_dataloader = create_dataloaders(
        tokenizer, 
        valid_data, 
        max_length, 
        batch_size, 
        property_value=property, 
        pooling=pooling
    )

    test_dataloader = create_dataloaders(
        tokenizer, 
        test_data, 
        max_length, 
        batch_size, 
        property_value=property, 
        pooling=pooling
    )

    # define the optimizer
    if optimizer_type == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr = learning_rate
        )
    elif optimizer_type == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=learn_rate
        )

    # set up the scheduler
    total_training_steps = len(train_dataloader) * epochs 
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup( #get_linear_schedule_with_warmup
            optimizer,
            num_warmup_steps= warmup_steps, #steps_ratio*total_training_steps,
            num_training_steps=total_training_steps 
        )
    
    # from <https://github.com/usnistgov/alignn/blob/main/alignn/train.py>
    elif scheduler_type == 'onecycle': 
        steps_per_epoch = len(train_dataloader)
        # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            # pct_start=pct_start,
            pct_start=0.3,
        )
    
    elif scheduler_type == 'step':
         # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=warmup_steps
        )
    
    elif scheduler_type == 'lambda':
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )

    training_stats, validation_predictions = train(model, optimizer, scheduler, mae_loss_function, mae_loss_function, 
        epochs, train_dataloader, valid_dataloader, device, normalizer=normalizer_type)

    print("======= Evaluating on test set ========")
    best_model_path = f"checkpoints/{task_name}/best_checkpoint_for_{property}.pt" 
    best_model = T5Predictor(base_model, base_model_output_size, drop_rate=drop_rate)

    if torch.cuda.device_count() > 1:
        print("Testing on", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        print("No CUDA available! Testing on CPU!")
        model.to(device)

    if isinstance(best_model, nn.DataParallel):
        model.module.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)), strict=False)
    else:
        best_model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)), strict=False) 
        best_model.to(device)
    
    _, test_performance = evaluate(best_model, mae_loss_function, test_dataloader, train_labels_mean, train_labels_std, property , normalizer=normalizer_type)
    

    
