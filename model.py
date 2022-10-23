"""
ByT5 finetuning on materials property prediction using materials textual description 
"""
# Import packages
import torch
import torch.nn as nn

class ByT5Reggressor(nn.Module):
    def __init__(self, base_model, base_model_output_size, n_classes, regressor_type, drop_rate=0.1, freeze_base_model=False):
        super(ByT5Reggressor, self).__init__()
        D_in, D_out = base_model_output_size, n_classes ## This might change
        self.model = base_model

        if regressor_type == "linear":
            self.regressor = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(D_in, D_out)
            )
        elif regressor_type == "MLP":
            print("use a neural network based regressor")

    def forward(self, input_ids, attention_masks):
        hidden_states = self.model(input_ids, attention_masks)
        # print("hidden_states_shape = ", hidden_states.size())
        print(hidden_states)
        print("-"*20)

        last_hidden_state = hidden_states.last_hidden_state # [batch_size, input_length, D_in]-->ByT5_output
        print("last_hidden_state_shape = ", last_hidden_state.size())
        print(last_hidden_state)
        print("-"*20)

        input_embedding = torch.sum(last_hidden_state, 1) # [batch_size, D_in] --> getting the embedding of the output by summing up the embeddings of the output characters 
        print("input_embedding = ", input_embedding.size())
        print(input_embedding)
        print("-"*20)

        outputs = self.regressor(input_embedding) # [batch_size, D_out] -->Feed the regression model only the last  hidden state of ByT5 model
        print("outputs_shape = ", outputs.size())
        print(outputs)
        print("-"*20)

        return outputs

        
