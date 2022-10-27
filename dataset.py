"""
A function to prepare the dataloaders
"""
# Import packages
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import *

np.random.seed(42)

def extract_mat_id(dir):
    dir_split = dir.split("/")
    # dir_split = dir.split("\\")
    get_mat_id = dir_split[len(dir_split)-1].split(".")[0]
    return get_mat_id

def load_data(mat_prop_dir, mat_descr_dir):
    df_data = 0

    mat_ids_list = []
    mat_formula_list = []
    mat_formation_energy = []
    mat_energy_above_hull = []
    mat_energy_per_atom = []
    mat_is_stable = []
    mat_description = []

    mat_prop_dir_list = glob.glob(f"{mat_prop_dir}/*.json") 
    mat_descr_dir_list = glob.glob(f"{mat_descr_dir}/*.json")
    # print(len(mat_prop_dir_list), len(mat_descr_dir_list))
    if len(mat_prop_dir_list) == len(mat_descr_dir_list)-1:
        not_detected_mat_decr = []
        not_valid_json_ids = []

        for i in range(len(mat_prop_dir_list)):
            mat_prop_id = extract_mat_id(mat_prop_dir_list[i])
            mat_descr_id = extract_mat_id(mat_descr_dir_list[i])
            if mat_prop_id == mat_descr_id:
                mat_prop_json = mat_prop_dir_list[i] #.replace("\\","/")
                mat_descr_json = mat_descr_dir_list[i] #.replace("\\","/")

                if is_json(mat_descr_json)=="True":
                    mat_prop_dict = readJSON(mat_prop_json)
                    mat_descr_dict = readJSON(mat_descr_json)

                    if len(mat_descr_dict["data"][0]) != 2:
                        not_detected_mat_decr.append(mat_descr_id)
                        continue
                    else:
                        mat_ids_list.append(mat_prop_dict["material_id"])
                        mat_formula_list.append(mat_prop_dict["property"]["formula_pretty"])
                        mat_formation_energy.append(mat_prop_dict["property"]["formation_energy_per_atom"])
                        mat_energy_above_hull.append(mat_prop_dict["property"]["energy_above_hull"])
                        mat_energy_per_atom.append(mat_prop_dict["property"]["energy_per_atom"])
                        mat_is_stable.append(mat_prop_dict["property"]["is_stable"])
                        mat_description.append(mat_descr_dict["data"][0]["description"])
                else:
                    not_valid_json_ids.append(mat_descr_id)
            else:
                continue

        df_data = pd.DataFrame(
            {
                "material_id": mat_ids_list,
                "formula_pretty": mat_formula_list,
                "formation_energy_per_atom": mat_formation_energy,
                "energy_above_hull": mat_energy_above_hull,
                "energy_per_atom": mat_energy_per_atom,
                "is_stable": mat_is_stable,
                "description": mat_description
            }
        )
        # writeTEXT(not_valid_json_ids, "statistics/formation_energy_not_valid_json_ids.txt")
        # writeTEXT(not_detected_mat_decr, "statistics/formation_energy_not_detected_mat_decr.txt")
        
        print(f"not_detected_mat_decr = {not_detected_mat_decr}", len(not_detected_mat_decr))
        print(f"not_valid_json_ids = {not_valid_json_ids}", len(not_valid_json_ids))
    else:
        print("Directory that contains material property do not match with the directory that contains material description")
    
    return df_data

def train_valid_test_split(prop_data_dir, mat_prop_dir, mat_descr_dir, split_ratio=[7,2,1]):
    if len(glob.glob(f"{prop_data_dir}/*.csv")) != 0:
        train_data = pd.read_csv(f"{prop_data_dir}/train.csv")
        valid_data = pd.read_csv(f"{prop_data_dir}/valid.csv")
        test_data = pd.read_csv(f"{prop_data_dir}/test.csv")
    else:
        df_data_all = load_data(mat_prop_dir, mat_descr_dir)
        train_ratio, valid_ratio, test_ratio = tuple([int((i/10)*len(df_data_all)) for i in split_ratio])

        ixs = np.arange(df_data_all.shape[0])
        np.random.shuffle(ixs) # randomly shuffle the index

        train_df_list = []
        for ix in ixs[0:train_ratio]:
            train_df_list.append(df_data_all.loc[[ix]])
        train_data = pd.concat(train_df_list, ignore_index=True)

        valid_df_list = []
        for ix in ixs[train_ratio:train_ratio+valid_ratio]:
            valid_df_list.append(df_data_all.loc[[ix]])
        valid_data = pd.concat(valid_df_list, ignore_index=True)

        test_df_list = []
        for ix in ixs[train_ratio+valid_ratio:train_ratio+valid_ratio+test_ratio]:
            test_df_list.append(df_data_all.loc[[ix]])
        test_data = pd.concat(test_df_list, ignore_index=True)

        saveCSV(train_data, f"{prop_data_dir}/train.csv")
        saveCSV(valid_data, f"{prop_data_dir}/valid.csv")
        saveCSV(test_data, f"{prop_data_dir}/test.csv")

    return train_data, valid_data, test_data

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

def create_dataloaders(tokenizer, dataframe, max_length, batch_size, property_value):
    """
    Dataloader which arrange the input sequences, attention masks, and labels in batches
    and transform the to tensors
    """
    input_ids, attention_masks = tokenize(tokenizer, dataframe, max_length)
    labels = dataframe.property_value.to_numpy()

    input_tensor = torch.tensor(input_ids)
    mask_tensor = torch.tensor(attention_masks)
    labels_tensor = torch.tensor(labels)

    dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # Set the shuffle to False for now since the labes are continues values check later if this may affect the result

    return dataloader
