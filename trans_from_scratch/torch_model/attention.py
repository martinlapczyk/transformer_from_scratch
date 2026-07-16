import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads==0
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        self.q_proj=nn.Linear(d_model, d_model)
        self.k_proj=nn.Linear(d_model, d_model)
        self.v_proj=nn.Linear(d_model, d_model)
        self.out_proj=nn.Linear(d_model, d_model)
        self.dropout=nn.Dropout()
    
    def split_heads(self,t,batch,seq_len):
        return t.view(batch,seq_len,self.num_heads,self.d_k).transpose(1,2)
    
    def forward(self,query_input,key_value_input,mask=None):
        batch,seq_len, _=query_input.shape
        seq_len_k=key_value_input.shape[1]

        q=self.split_heads(self.q_proj(query_input),batch,seq_len_k)
        k=self.split_heads(self.k_proj(key_value_input),batch,seq_len_k)
        v=self.split_heads(self.v_proj(key_value_input),batch,seq_len_k)

        scores=(q@k.transpose(-2,-1))/math.sqrt(self.d_k)
        if mask is not None:
            scores=scores.masked_fill(mask==0,float('-inf'))
        weights=self.dropout(torch.softmax(scores,dim=-1))

        out=weights@v
        out=out.transpose(1,2).contiguous().view(batch,seq_len_q, self.num_heads*self.d_k)
        return self.out_proj(out), weights