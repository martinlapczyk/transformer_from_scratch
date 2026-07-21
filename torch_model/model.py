import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from masks import make_src_mask, make_causal_mask
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128,num_heads=4,num_layers=3,d_ff=512,max_len=128,dropout=0.1,pad_idx=0):
        super().__init__()
        self.encoder=Encoder(src_vocab_size,d_model,num_heads,num_layers,d_ff,max_len,dropout)
        self.decoder=Decoder(tgt_vocab_size,d_model,num_heads,num_layers,d_ff,max_len,dropout)
        self.out_proj=nn.Linear(d_model,tgt_vocab_size)
        self.pad_idx=pad_idx
    
    def make_src_mask(self,src):
        return make_src_mask(src,self.pad_idx)
    
    def make_causal_mask(self,tgt):
        return make_causal_mask(tgt,self.pad_idx)
    
    def forward(self,src,tgt):
        src_mask=self.make_src_mask(src)
        causal_mask=self.make_causal_mask(tgt)
        cross_mask=src_mask

        encoder_out=self.encoder(src,src_mask)
        decoder_out,cross_weights=self.decoder(tgt,encoder_out,causal_mask,cross_mask)
        logits=self.out_proj(decoder_out)
        return logits,cross_weights
    