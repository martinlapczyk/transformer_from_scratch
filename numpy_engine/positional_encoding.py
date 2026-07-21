import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(seq_len,d_model):
    pos=np.arange(seq_len)[:, None]
    i=np.arange(d_model)[None,:]
    angle_rates=1/np.power(10000, (2*(i//2))/np.float32(d_model))
    angles=pos*angle_rates

    pe=np.zeros((seq_len,d_model))
    pe[:,0::2]=np.sin(angles[:,0::2])
    pe[:,1::2]=np.cos(angles[:,1::2])
    return pe
def plot_pe(pe):
    plt.imshow(pe,cmap='viridis')
    plt.colorbar()
    plt.show()
pe=positional_encoding(seq_len=50, d_model=128)
plot_pe(pe)