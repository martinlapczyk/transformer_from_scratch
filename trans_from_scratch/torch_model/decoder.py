class DecoderBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super().__init__()
        self.self_attn=MultiHeadAttention(d_model,num_heads,dropout)
        self.cross_attn=MultiHeadAttention(d_model,num_heads,dropout)
        self.ff=FeedForward(d_model,d_ff,dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x,encoder_out,causal_mask=None,cross_mask=None):
        self_attn_out, _=self.self_attn(x,x,mask=causal_mask)
        x=self.norm1(x+self.dropout(self_attn_out))

        cross_attn_out, _=self.cross_attn(x, encoder_out,mask=cross_mask)
        x=self.norm2(x+self.dropout(cross_attn_out))

        ff_out=self.ff(x)
        x=self.norm3(x+self.dropout(ff_out))
        return x, cross_weights
    
class Decoder(nn.Module):
    def __init__(self,vocab_size,d_model,num_heads,num_layers,d_ff,max_len,dropout=0.1):
        super().__init__()
        self.token_emb=nn.Embedding(vocab_size,d_model)
        self.pos_enc=PositionalEncoding(d_model,max_len)
        self.blocks=nn.ModuleList([
            DecoderBlock(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)
        ])
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,tgt,encoder_out,causal_mask=None,cross_mask=None):
        x=self.dropout(self.pos_enc(self.token_emb(tgt)))
        cross_weights_all=[]
        for block in self.blocks:
            x,cross_weights=block(x,encoder_out,causal_mask,cross_mask)
            cross_weights_all.append(cross_weights)
        return x, cross_weights_all