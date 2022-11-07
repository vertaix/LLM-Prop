"""
Codes to evaluate the trained models onn held out test set
"""
# Import necessary packages
import time
import torch
import torch.nn as nn
import pandas as pd

from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from utils import *
from dataset import create_dataloaders

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
    testing_time = time_format(test_ending_time-test_start_time)

    test_predictions = {f"{property_value}":predictions_list}

    return test_predictions, average_test_loss, testing_time

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
    data_path = f"data/property/{property_name}/{property_value}/test.csv"
    max_length = 512
    regressor_type = "linear"  # can also be mlp
    dataframe = pd.read_csv(data_path)

    # Define the loss functions: using "mean absolute error:mae" and "mean square error:mse" losses
    loss_type = "mae"
    if loss_type == "mae":
        loss_function = nn.L1Loss()
    elif loss_type == "mse":
        loss_function = nn.MSELoss()

    model_names = ["t5-small"]
    for model_name in model_names:
        if model_name == "t5-small":
            batch_size = 64
            epochs = 30
            tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-small")
        elif model_name == "t5-base":
            batch_size = 16
            epochs = 50
            tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base")
        
        model_path = f"model_checkpoints/{property_name}/{model_name}/{model_name}_finetuned_{regressor_type}_using_{loss_type}_loss_with_{epochs}_epochs.pt"
        model = torch.load(model_path, map_location=torch.device(device))
        test_dataloader = create_dataloaders(tokenizer, dataframe, max_length, batch_size)

        test_predictions, average_test_loss, testing_time = evaluate(model, test_dataloader, loss_function, property_value)

        print(f"Testing time took {testing_time}")
        print(f"Test loss for {model_name} is {average_test_loss}")
        saveCSV(pd.DataFrame(test_predictions), f"statistics/{property_name}/{model_name}/test_statistics_for_{model_name}_finetuned_{regressor_type}_using_{loss_type}_loss_after_{epochs}_epochs.csv")