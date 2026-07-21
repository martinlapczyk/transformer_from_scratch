import numpy as np
from attention import MultiHeadAttention
from layers import layer_norm, feedforward

class TransformerBlock:
    def __init__(self,d_model,num_heads,d_ff,seed=0):
        self.attn=MultiHeadAttention(d_model,num_heads,seed=seed)

        rng=np.random.default_rng(seed+1)
        scale=np.sqrt(1/d_model)
        self.W1=rng.normal(0,scale,(d_model,d_ff))
        self.b1=np.zeros(d_ff)
        self.W2=rng.normal(0,scale,(d_ff,d_model))
        self.b2=np.zeros(d_model)

    def forward(self,x,mask=None):
        attn_out,weights=self.attn.forward(x,mask=mask)
        x=layer_norm(x+attn_out)
        ff_out=feedforward(x,self.W1,self.b1,self.W2,self.b2)
        x=layer_norm(x+ff_out)
        return x,weights