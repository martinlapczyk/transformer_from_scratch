import torch
import torch.nn as nn
import math
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d_model,d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff,d_model)
        )
    def forward(self,x):
        return self.net(x)
    
class PositionalEncoding(nn.Module):
    def __init__ (self, d_model, max_len=5000):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(0,max_len).unsqueeze(1).float()
        div_term=torch.exp(torch.arange(0,d_model,2).float()*-(math.log(10000.0)/d_model))
        pe[:, 0::2]=torch.sin(pos*div_term)
        pe[:, 1::2]=torch.cos(pos*div_term)
        self.register_buffer('pe',pe.unsqueeze(0))

    def forward(self,x):
        return x+self.pe[:, :x.size(1)]