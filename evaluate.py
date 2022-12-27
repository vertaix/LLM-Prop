"""
Codes to evaluate the trained models on held out test set
"""
# Import necessary packages
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from statistics import stdev

from utils import *
from dataset import create_dataloaders
from model import T5Reggressor

def evaluate(model, test_dataloader, loss_function, property_value):
    test_start_time = time.time()

    model.eval()

    total_test_loss = 0
    predictions_list = []

    for step, batch in enumerate(test_dataloader):
        batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            predictions = model(batch_inputs, batch_masks)
            loss = loss_function(predictions.squeeze(), batch_labels.squeeze()) 
        total_test_loss += loss.item()
        predictions = predictions.detach().cpu().numpy()
        for i in predictions:
            predictions_list.append(i[0])

    average_test_loss = total_test_loss / len(test_dataloader)
    test_ending_time = time.time()
    # testing_time = time_format(test_ending_time-test_start_time)

    # test_predictions = {f"{property_value}":predictions_list}

    return predictions_list, average_test_loss, test_ending_time-test_start_time

if __name__=="__main__":

    # Check if the GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Number of available devices: {torch.cuda.device_count()}')
        print(f'Current device is: {torch.cuda.current_device()}')
    else:
        print("No GPU available, please connect to the GPU first or continue to use CPU instead")
        device = torch.device("cpu")

    property_name = "formation_energy"
    property_value = "formation_energy_per_atom"
    data_path = f"data/property/{property_name}/{property_value}"
    max_length = 512
    regressor_type = "linear"  # can also be mlp
    dataframe_test = pd.read_csv(f"{data_path}/test.csv")
    dataframe_matbench = pd.read_csv(f"{data_path}/test_matbench.csv")

    # arguments for the pretrained model
    batch_size = 64
    max_length = 512
    hidden_dim = 128
    n_layers = 2
    n_filters = 100
    filter_sizes = [3,4,5]
    drop_rate = 0.1
    base_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small")
    base_model_output_size = 512

    # Define the loss functions: using "mean absolute error:mae" and "mean square error:mse" losses
    loss_type = "mae"
    if loss_type == "mae":
        loss_function = nn.L1Loss()
    elif loss_type == "mse":
        loss_function = nn.MSELoss()

    model_paths = [ 
       f"model_checkpoints/{property_name}/t5-small/ablation/t5-small_finetuned_linear_using_mae_loss_with_60_epochs_on_20447_examples.pt",
       f"model_checkpoints/{property_name}/t5-small/ablation/t5-small_finetuned_linear_using_mae_loss_with_60_epochs_on_40894_examples.pt",
       f"model_checkpoints/{property_name}/t5-small/ablation/t5-small_finetuned_linear_using_mae_loss_with_60_epochs_on_61341_examples.pt",
       f"model_checkpoints/{property_name}/t5-small/ablation/t5-small_finetuned_linear_using_mae_loss_with_60_epochs_on_81788_examples.pt",
       f"model_checkpoints/{property_name}/t5-small/ablation/normalized_t5-small_finetuned_linear_using_mae_loss_with_60_epochs.pt"
    ]
    batch_size = 64
    testing_duration = []
    test_loss = []

    for dataframe in [dataframe_test, dataframe_matbench]:
        if "matbench" in dataframe:
            test_name = "matbench"
        else:
            test_name = "test"

        for i, model_path in enumerate(model_paths):
            # if "baseline" in model_path:
            #     tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-small")
            #     model_type = "baseline"
            #     best_epoch = 46
            # else :
            #     tokenizer = AutoTokenizer.from_pretrained("tokenizers/new_pretrained_t5_tokenizer_on_modified_c4_and_mat_descriptions_52k_vocab")
            #     base_model.resize_token_embeddings(len(tokenizer))
            #     model_type = "proposed"
            #     best_epoch = 52
            
            tokenizer = AutoTokenizer.from_pretrained("tokenizers/new_pretrained_t5_tokenizer_on_modified_c4_and_mat_descriptions_52k_vocab")
            base_model.resize_token_embeddings(len(tokenizer))

            model =  T5Reggressor(base_model, base_model_output_size, regressor_type, hidden_dim, filter_sizes, n_layers, n_filters, drop_rate=drop_rate)
            model.load_state_dict(torch.load(model_path, map_location=torch.device(device)), strict=False)
            test_dataloader = create_dataloaders(tokenizer, dataframe, max_length, batch_size)

            predictions = []
            test_losses = []
            test_durations = []
            for i in range(5):
                predictions_list, average_test_loss, testing_time = evaluate(model, test_dataloader, loss_function, property_value)
                predictions.append(predictions_list)
                test_losses.append(average_test_loss)
                test_durations.append(testing_time)
            
            averaged_predictions = np.mean(np.array(predictions), axis=0)
            averaged_loss = np.mean(test_losses)
            averaged_test_time = np.sum(test_durations)

            confidence_score = stdev(test_losses)

            testing_duration.append(time_format(averaged_test_time))
            test_loss.append(averaged_loss)
            test_predictions = {f"{property_value}":averaged_predictions}

            print(f"Testing time for {test_name} took {testing_duration}")
            print(f"Test loss for {test_name} for the ablation model_{i+1} is {averaged_loss}")
            print(f"Confidence score for {test_name} for the ablation model_{i+1} is {confidence_score}")
            print("="*100)

            saveCSV(pd.DataFrame(test_predictions), f"statistics/test/{property_name}/t5-small/ablation_model_{i+1}_{test_name}_statistics_for_t5_small_finetuned_{regressor_type}_using_{loss_type}_loss.csv")
        
        writeTEXT(testing_duration, f"statistics/test/{property_name}/t5-small/ablation_model_{i+1}_{test_name}_duration_for_t5_small_finetuned_{regressor_type}_using_{loss_type}.txt")
        writeTEXT(test_loss, f"statistics/test/{property_name}/t5-small/ablation_model_{i+1}_{test_name}_loss_for_t5_small_finetuned_{regressor_type}_using_{loss_type}.txt")