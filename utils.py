import json
import datetime

def is_json(json_file):
    """
    Checks whether a json file is valid
    """
    with open(json_file, "r", encoding="utf-8") as f:
        sent = f.read().split(":")
        if len(sent)>=3 and len(sent) < 6:
            dec = "True"
        else:
            dec = "False"
    return dec

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

