import json
import datetime
import torch
from sklearn.metrics import roc_auc_score

def writeToJSON(data, where_to_save):
    """
    data: a dictionary that contains data to save
    where_to_save: the name of the file to write on
    """
    with open(where_to_save, "w", encoding="utf8") as outfile:
        json.dump(data, outfile)

def readJSON(input_file):
    """
    1. arguments
        input_file: a json file to read
    2. output
        a json objet in a form of a dictionary
    """
    with open(input_file, "r", encoding="utf-8", errors='ignore') as infile:
        json_object = json.load(infile, strict=False)
    return json_object

def writeTEXT(data, where_to_save):
    with open(where_to_save, "w", encoding="utf-8") as outfile:
        for d in data:
            outfile.write(str(d))
            outfile.write("\n")

def readTEXT_to_LIST(input_file):
    with open(input_file, "r", encoding="utf-8") as infile:
        data = []
        for line in infile:
            data.append(line)
    return data

def saveCSV(df, where_to_save):
    df.to_csv(where_to_save, index=False)

def time_format(total_time):
    """
    Change the from seconds to hh:mm:ss
    """
    total_time_rounded = int(round((total_time)))
    total_time_final = str(datetime.timedelta(seconds=total_time_rounded))
    return total_time_final

def z_normalizer(labels):
    """ Implement a z-score normalization technique"""
    labels_mean = torch.mean(labels)
    labels_std = torch.std(labels)

    scaled_labels = (labels - labels_mean) / labels_std

    return scaled_labels

def z_denormalize(scaled_labels, labels_mean, labels_std):
    labels = (scaled_labels * labels_std) + labels_mean
    return labels

def min_max_scaling(labels):
    """ Implement a min-max normalization technique"""
    min_val = torch.min(labels)
    max_val = torch.max(labels)
    diff = max_val - min_val
    scaled_labels = (labels - min_val) / diff
    return scaled_labels

def mm_denormalize(scaled_labels, min_val, max_val):
    diff = max_val - min_val
    denorm_labels = (scaled_labels * diff) + min_val
    return denorm_labels

def log_scaling(labels):
    """ Implement log-scaling normalization technique"""
    scaled_labels = torch.log1p(labels)
    return scaled_labels

def ls_denormalize(scaled_labels):
    denorm_labels = torch.expm1(scaled_labels)
    return denorm_labels
