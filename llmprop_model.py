"""
T5 finetuning on materials property prediction using materials text description 
"""
# Import packages
import torch
import torch.nn as nn
import torch.nn.functional as F

class T5Predictor(nn.Module):
    def __init__(
        self, 
        base_model, 
        base_model_output_size, 
        n_classes=1, 
        drop_rate=0.1,  
        pooling='cls'
    ):
        super(T5Predictor, self).__init__()
        D_in, D_out = base_model_output_size, n_classes
        self.model = base_model
        self.dropout = nn.Dropout(drop_rate)
        self.pooling = pooling

        # instantiate a linear regressor
        self.linear_regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out)
        )

    def forward(self, input_ids, attention_masks):
        
        hidden_states = self.model(input_ids, attention_masks)

        last_hidden_state = hidden_states.last_hidden_state # [batch_size, input_length, D_in]
        
        if self.pooling == 'cls':
            input_embedding = last_hidden_state[:,0,:] # [batch_size, D_in] -- [CLS] pooling
        elif self.pooling == 'mean':
            input_embedding = last_hidden_state.mean(dim=1) # [batch_size, D_in] -- mean pooling
        
        outputs = self.linear_regressor(input_embedding) # [batch_size, D_out]

        return input_embedding, outputs