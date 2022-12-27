"""
A function to prepare the dataloaders that read the whole input sequence chunk by chunk
"""
# Import packages
import glob
import pandas as pd

def train_valid_test_split(prop_data_dir):
    if len(glob.glob(f"{prop_data_dir}/*.csv")) != 0:
        train_data = pd.read_csv(f"{prop_data_dir}/train.csv")
        valid_data = pd.read_csv(f"{prop_data_dir}/valid.csv")
        test_data = pd.read_csv(f"{prop_data_dir}/test.csv")
    else:
        print(f"No training data found at {prop_data_dir}!")

    return train_data, valid_data, test_data

def custom_dataloader(tokenizer, dataframe, chunksize, batch_size, order="descending"):
    """
    A dataloader which 
    1. reorders (descending, ascending) the input to make similar input in length to be close to each other 
    2. pads the input sequence by batch following the longest input sequence in that batch
    3. splits the input into multiple chunks with the same size (chunk_size) 
    4. arranges the input sequences, attention masks, and labels in batches
    and transform them to tensors 
    5. returns a list of the data (input_ids, attn_masks, labels) in tensors for each batch
    """
    # get the lenght of each input sequence
    df_list = [len(tokenizer.tokenize(dataframe['description'][i])) for i in range(len(dataframe))]
    dataframe['length'] = df_list

    # reorder the input
    if order=='descending':
        dataframe.sort_values(by='length', ascending=False)
    elif order=="ascending":
        dataframe.sort_values(by='length', ascending=True) 
    elif order=='none':
        dataframe = dataframe

    labels = dataframe.formation_energy_per_atom.to_numpy()
    inputs = dataframe.description.tolist()
    batch_data = []

    t=0
    while t < len(inputs):
        input_ids = []
        attention_mask = []

        # split the input samples into batches
        temp_bs_lst = inputs[t:t+batch_size]
        max_len_per_batch = max([len(tokenizer.tokenize(descr)) for descr in temp_bs_lst])
        
        # pad the input sequence in each batch by the longest input sequence in that batch
        tokens = tokenizer(temp_bs_lst, 
                        add_special_tokens=False,
                        return_attention_mask=True,
                        padding='max_length',
                        truncation='longest_first',
                        max_length=max_len_per_batch,
                        return_tensors='pt')

        input_id_chunks = [tokens['input_ids'][i].split(chunksize) for i in range(len(tokens['input_ids']))]
        mask_chunks = [tokens['attention_mask'][i].split(chunksize) for i in range(len(tokens['attention_mask']))]

        temp_input_id_chunks = input_id_chunks
        temp_mask_chunks = mask_chunks

        # loop through each chunk
        for i in range(len(temp_input_id_chunks)):
            temp_id_chunks = list(temp_input_id_chunks[i])
            temp_msk_chunks = list(temp_mask_chunks[i])

            # get required padding length
            for j in range(len(temp_id_chunks)):
                pad_len = chunksize - temp_id_chunks[j].shape[0]
                
                # check if tensor length satisfies required chunk size
                if pad_len > 0:
                    # if padding length is more than 0, we must add padding
                    temp_id_chunks[j] = torch.cat([
                        temp_id_chunks[j], torch.Tensor([0] * pad_len)
                    ])

                    temp_msk_chunks[j] = torch.cat([
                        temp_msk_chunks[j], torch.Tensor([0] * pad_len)
                    ])

            temp_input_id_chunks[i] = tuple(temp_id_chunks)
            temp_mask_chunks[i] = tuple(temp_msk_chunks)

        input_id_chunks = tuple(temp_input_id_chunks)
        mask_chunks = tuple(temp_mask_chunks)

        for i in range(len(input_id_chunks)):
            input_ids.append(torch.stack(input_id_chunks[i]).long())
            attention_mask.append(torch.stack(mask_chunks[i]).int())

            input_dict = {
                'input_ids': tuple(input_ids),
                'attention_mask': tuple(attention_mask)
            }

    input_ids = input_dict["input_ids"]
    attention_masks = input_dict['attention_mask']

    input_tensor = torch.stack(list(input_ids), dim=0)
    attention_tensor = torch.stack(list(attention_masks), dim=0)
    labels_tensor = torch.tensor(labels[t:t+batch_size])

    batch_data.append({'input_tensor':input_tensor, 
                        'mask_tensor':attention_tensor, 
                        'label_tensor':labels_tensor})

    t += batch_size

    return batch_data