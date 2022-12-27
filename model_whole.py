"""
T5 finetuning on materials property prediction using materials textual description 
"""
# Import packages
import torch
import torch.nn as nn
import torch.nn.functional as F

class T5Reggressor(nn.Module):
    def __init__(self, base_model, base_model_output_size, regressor_type, hidden_dim, filter_sizes, n_layers, n_filters, n_classes=1, drop_rate=0.1, freeze_base_model=False, bidirectional=True):
        super(T5Reggressor, self).__init__()
        D_in, D_out = base_model_output_size, n_classes # n_classes should always equal to one for regression
        self.model = base_model
        self.regressor = regressor_type
        self.dropout = nn.Dropout(drop_rate)

        # instantiate a linear regressor
        self.linear_regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out)
        )

        # instantiate a multilayer perceptron (mlp) regressor
        self.fc_1 = nn.Linear(D_in, 256)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_3 = nn.Linear(128, 64)
        self.fc_last = nn.Linear(64, D_out)
        self.relu = nn.ReLU()

        # instantiate a recurrent neural network (rnn) regressor
        self.rnn = nn.GRU(D_in, hidden_dim, n_layers, bidirectional=bidirectional, dropout=drop_rate, batch_first=True)
        self.rnn_fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, D_out)
        
        # instantiate a convolution neural network (cnn) regressor
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels = 1,
                out_channels = n_filters,
                kernel_size = (fs, D_in))
            for fs in filter_sizes
        ])
        self.cnn_fc = nn.Linear(len(filter_sizes) * n_filters, D_out)

    def forward(self, input_ids, attention_masks):
        input_ids_list = torch.unbind(input_ids, dim=0) # input_ids: [bsz, chunk_nbrs, chunk_len] => [chunk_nbrs, chunk_len]*bsz
        attention_mask_list = torch.unbind(attention_masks, dim=0) # attention_masks: [bsz, chunks_nbrs, chunk_len] => [chunk_nbrs, chunk_len]*bsz
        hidden_states = (self.model(input_ids_list[i], attention_mask_list[i]) for i in range(len(input_ids_list)))
        last_hidden_states = (hidden_state.last_hidden_state for hidden_state in hidden_states) # [chunk_nbrs, chunk_len, D_in]*bsz
        mean_last_hidden_states = (last_hidden_state.mean(dim=0) for last_hidden_state in last_hidden_states) # [chunk_len, D_in]*bsz
        last_hidden_state = torch.stack(list(mean_last_hidden_states)) # [bsz, chunk_len, D_in]

        if self.regressor == "linear":
            """using a linear regressor"""
            input_embedding = last_hidden_state.mean(dim=1) # [batch_size, D_in] --> getting the embedding of the output by averaging the embeddings of the output characters 
            outputs = self.linear_regressor(input_embedding) # [batch_size, D_out] -->Feed the regression model only the last  hidden state of T5 model

        elif self.regressor == "mlp":
            input_embedding = self.dropout(last_hidden_state)
            output_1 =self.relu(self.fc_1(input_embedding))
            output_2 = self.relu(self.fc_2(self.dropout(output_1)))
            output_3 = self.relu(self.fc_3(self.dropout(output_2))) 
            outputs = self.fc_last(output_3)
            
        elif self.regressor == "gru":
            """using an rnn-based regressor"""
            input_embedding = self.dropout(last_hidden_state) # [batch_size, input_length, D_in]
            output, hidden_state = self.rnn(input_embedding) # output = [batch_size, input_len, hid dim * num directions], hidden_state = [num layers * num directions, batch size, hid dim]
            hidden = self.dropout(torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)) # [batch_size, hid dim * num directions]
            outputs = self.rnn_fc(hidden) # [batch_size, D_out]

        elif self.regressor == "cnn":
            """using a cnn-based regressor"""
            input_embedding = last_hidden_state.unsqueeze(1) # [batch_size, 1, input_length, D_in]
            conved = [F.relu(conv(input_embedding)).squeeze(3) for conv in self.convs] # conv_i = [batch_size, n_filters, input_len - filter_sizes[i]]
            pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] # pooled_i = [batch_size, n_filters]
            concat = self.dropout(torch.cat(pooled, dim=1)) # [batch_size, n_filters * len(filter_sizes)]
            outputs = self.cnn_fc(concat) # [batch_size, D_out]

        return outputs