import numpy as np

def layer_norm(x, eps=1e-6):
    mean=x.mean(axis=-1, keepdims=True)
    std=x.std(axis=-1, keepdims=True)
    return (x-mean)/(std+eps)

def feedforward(x,W1,b1,W2,b2):
    hidden=np.maximum(0,x@W1+b1)
    return hidden@W2+b2

class Transformer:
    def __init__(self, d_model, num_heads, d_ff, seed=0):
        self.attn=MultiHeadAttention(d_model, num_heads, seed=seed)
        rng=np.random.default_rng(seed+1)
        scale=np.sqrt(1/d_model)
        self.W1=rng.normal(0,scale,(d_model,d_ff))
        self.b1=np.zeros(d_ff)
        self.W2=rng.normal(0,scale,(d_ff,d_model))
        self.b2=np.zeros(d_model)

    def forward(self, x, mask=None):
        attn_out, weights=self.attn.forward(x,mask=mask)
        x=layer_norm(x+attn_out)
        ff_out=feedforward(x,self.W1,self.b1, self.W2, self.b2)
        x=layer_norm(x+ff_out)
        return x, weights
    
    