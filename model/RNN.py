# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/042_models.RNN.ipynb.

# %% auto 0
__all__ = ['RNN', 'LSTM', 'GRU']

# %% ../../nbs/042_models.RNN.ipynb 3
from model.imports import *
from model.utils import *
from model.layers import *
from model.utils_2 import *
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from torch.nn.utils import weight_norm

# %% ../../nbs/042_models.RNN.ipynb 4
class _RNN_Base(Module):
    def __init__(self, c_in, c_out, hidden_size=100, n_layers=1, bias=True, rnn_dropout=0, bidirectional=False, fc_dropout=0., init_weights=True):
        self.rnn = self._cell(c_in, hidden_size, num_layers=n_layers, bias=bias, batch_first=True, dropout=rnn_dropout, 
                              bidirectional=bidirectional)
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else nn.Identity()
        self.fc = nn.Linear(hidden_size * (1 + bidirectional), c_out)
        if init_weights: self.apply(self._weights_init)

    def forward(self, x): 
        x = x.transpose(2,1)    # [batch_size x n_vars x seq_len] --> [batch_size x seq_len x n_vars]
        output, _ = self.rnn(x) # output from all sequence steps: [batch_size x seq_len x hidden_size * (1 + bidirectional)]
        output = output[:, -1]  # output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        output = self.fc(self.dropout(output))
        return output
    
    def _weights_init(self, m): 
        # same initialization as keras. Adapted from the initialization developed 
        # by JUN KODA (https://www.kaggle.com/junkoda) in this notebook
        # https://www.kaggle.com/junkoda/pytorch-lstm-with-tensorflow-like-initialization
        for name, params in m.named_parameters():
            if "weight_ih" in name: 
                nn.init.xavier_normal_(params)
            elif 'weight_hh' in name: 
                nn.init.orthogonal_(params)
            elif 'bias_ih' in name:
                params.data.fill_(0)
                # Set forget-gate bias to 1
                n = params.size(0)
                params.data[(n // 4):(n // 2)].fill_(1)
            elif 'bias_hh' in name:
                params.data.fill_(0)
        
class RNN(_RNN_Base):
    _cell = nn.RNN
    
class LSTM(_RNN_Base):
    _cell = nn.LSTM
    
class GRU(_RNN_Base):
    _cell = nn.GRU
