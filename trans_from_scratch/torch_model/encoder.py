class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn=MultiHeadAttention(d_model, num_heads, dropout)
        self.ff=FeedForward(d_model, d_ff, dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out, _=self.self_attn(x,x,mask=src_mask)
        x=self.norm1(x+self.dropout(attn_out))
        ff_out=self.ff(x)
        x=self.norm2(x+self.dropout(ff_out))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size,d_model,num_heads,num_layers,d_ff,max_len,dropout=0.1):
        super().__init__()
        self.token_emb=nn.Embedding(vocab_size,d_model)
        self.pos_enc=PositionalEncoding(d_model,max_len)
        self.blocks=nn.ModuleList([
            EncoderBlock(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)
        ])
        self.dropout=nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        x=self.dropout(self.pos_enc(self.token_emb(src)))
        for block in self.blocks:
            x=block(x,src_mask)
        return x
    