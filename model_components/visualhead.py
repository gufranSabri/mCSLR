"""
This file is modified from:
https://github.com/sutwangyan/MSKA
"""


import torch
import torch.nn.functional as F
from model_components.utils import PositionalEncoding, MaskedNorm, PositionwiseFeedForward
from model_components.moe import MoE

class VisualHead(torch.nn.Module):
    def __init__(self, cls_num, input_size=512, hidden_size=1024, ff_size=2048, ff_kernelsize=[3,3]):
        super().__init__()

        self.hidden_size = hidden_size

        self.fc1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.bn1 = MaskedNorm(num_features=self.hidden_size, norm_type='batch')
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=0.1)

        self.pe = PositionalEncoding(self.hidden_size)

        # self.feedforward = PositionwiseFeedForward(
        #     input_size=self.hidden_size, ff_size=ff_size,
        #     dropout=0.1, kernel_size=ff_kernelsize, skip_connection=True
        # )
        self.feedforward = MoE(
            input_dim=self.hidden_size, 
            ff_size=ff_size,
            ff_kernelsize=ff_kernelsize, 
            num_experts=4,
            top_k=2

        )
        
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.gloss_output_layer = torch.nn.Linear(self.hidden_size, cls_num)


    def forward(self, x, mask, valid_len_in=None):
        B, Tin, D = x.shape 

        #projection 1
        x = self.fc1(x)
        x = self.bn1(x, mask)
        x = self.relu1(x)
        
        #pe
        x = self.pe(x)
        x = self.dropout1(x)

        #feedforward
        x, aux_loss = self.feedforward(x)
        x = self.layer_norm(x)

        x = x.transpose(1,2)
        x = x.transpose(1,2)

        #classification
        logits = self.gloss_output_layer(x) #B,T,V
        gloss_probabilities_log = logits.log_softmax(2) 
        gloss_probabilities = logits.softmax(2)

        valid_len_out = valid_len_in

        return {
            'gloss_feature': x,
            'gloss_feature_norm': F.normalize(x, dim=-1),
            'gloss_logits':logits, 
            'gloss_probabilities_log':gloss_probabilities_log,
            'gloss_probabilities': gloss_probabilities,
            'valid_len_out':valid_len_out,
            'aux_loss': aux_loss
        }