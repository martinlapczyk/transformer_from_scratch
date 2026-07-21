import torch
def make_src_mask(src,pad_idx):
    return (src!=pad_idx).unsqueeze(1).unsqueeze(2)

def make_causal_mask(tgt,pad_idx):
    batch,seq_len=tgt.shape
    pad_mask=(tgt!=pad_idx).unsqueeze(1).unsqueeze(2)
    causal=torch.tril(torch.ones(seq_len,seq_len,device=tgt.device)).bool()
    return pad_mask&causal
