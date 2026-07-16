import numpy as np

def softmax(x, axis=-1):
    e_x=np.exp(x-np.max(x, axis=axis, keepdims=True))
    return e_x/e_x.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q,K,V,mask=None):
    """
    Q:(...,seq_len_q,d_q)
    K:(...,seq_len_k,d_k)
    V:(...,seq_len_v,d_v)
    mask: (..., seq_len_q,seq_len_k)
    """

    d_k=Q.shape[-1]
    scores=Q@np.swapaxes(K,-1,-2)/np.sqrt(d_k)

    if mask is not None:
        scores=np.where(mask==0,-1e9,scores)
    weights=softmax(scores,axis=-1)
    output=weights@V
    return output, weights

class MultiHeadAttention:
    def __init__(self,d_model,num_heads,seed=0):
        assert d_model%num_heads==0
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads

        rng=np.random.default_rng(seed)
        scale=np.sqrt(1/d_model)
        self.W_q=rng.normal(0,scale,(d_model,d_model))
        self.W_k=rng.normal(0,scale,(d_model,d_model))
        self.W_v=rng.normal(0,scale,(d_model,d_model))
        self.W_o=rng.normal(0,scale,(d_model,d_model))

        def split_heads(self,x):
            batch,seq_len, _=x.shape
            x=x.reshape(batch,seq_len,self.num_heads,self.d_k)
            return x.transpose(0,2,1,3)
        def combine_heads(self,x):
            batch,num_heads,seq_len,d_k=x.shape
            x=x.transpose(0,2,1,3)
            return x.reshape(batch,seq_len,num_heads*d_k)
        def forward(self,x,mask=None):
            Q=x@self.W_q
            K=x@self.W_k
            V=x@self.W_v

            Q,K,V=self.split_heads(Q), self.split_heads(K),self.split_heads(V)
            out,weights=scaled_dot_product_attention(Q,K,V,mask=mask)
            out=self.combine_heads(out)
            return out@self.W_o,weights

