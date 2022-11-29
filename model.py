"""
ByT5 finetuning on materials property prediction using materials textual description 
"""
# Import packages
import torch
import torch.nn as nn

class T5Reggressor(nn.Module):
    def __init__(self, base_model, base_model_output_size, n_classes, regressor_type, hidden_dim, n_layers, drop_rate=0.1, freeze_base_model=False, bidirectional=True):
        super(T5Reggressor, self).__init__()
        D_in, D_out = base_model_output_size, n_classes ## This might change
        self.model = base_model
        self.regressor = regressor_type

        if regressor_type == "linear":
            print("using a linear regressor")
            self.linear_regressor = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(D_in, D_out)
            )
        elif regressor_type == "rnn":
            print("using a recurrent neural network based regressor")
            self.dropout = nn.Dropout(drop_rate)
            # nn.utils.rnn.pack_padded_sequence(batch_first=True, enforce_sorted=False),
            self.rnn = nn.GRU(D_in, hidden_dim, n_layers, bidirectional=bidirectional, dropout=drop_rate, batch_first=True)
            # nn.utils.rnn.pack_padded_sequence(),
            self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, D_out)

    def forward(self, input_ids, attention_masks):
        hidden_states = self.model(input_ids, attention_masks)

        last_hidden_state = hidden_states.last_hidden_state # [batch_size, input_length, D_in]

        if self.regressor == "linear":
            input_embedding = torch.sum(last_hidden_state, 1)/last_hidden_state.size()[1] # [batch_size, D_in] --> getting the embedding of the output by averaging the embeddings of the output characters 
            outputs = self.linear_regressor(input_embedding) # [batch_size, D_out] -->Feed the regression model only the last  hidden state of T5 model
        elif self.regressor == "rnn":
            input_embedding = self.dropout(last_hidden_state) # [batch_size, input_length, D_in]
            output, hidden_state = self.rnn(input_embedding) # output = [batch size, input_len, hid dim * num directions], hidden_state = [num layers * num directions, batch size, hid dim]
            hidden = self.dropout(torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)) # [batch size, hid dim * num directions]
            outputs = self.fc(hidden) # [batch_size, D_out]

        return outputs

        
