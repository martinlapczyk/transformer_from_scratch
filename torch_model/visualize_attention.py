import matplotlib.pyplot as plt

def plot_cross_attention(cross_weights, src_tokens, tgt_tokens, layer=-1, head=0):
    w=cross_weights[layer][0,head].detach().cpu().numpy()
    fig,ax=plt.subplots(figsize=(8,6))
    im=ax.imshow(w,cmap='viridis')
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens,rotation=90)
    ax.set_yticklabels(tgt_tokens)
    ax.set_xlabel("Source (encoder)")
    ax.set_ylabel("Target (decoder,generated)")
    ax.set_title(f"Cross-attention-layer {layer}, head {head}")
    fig.colorbar(im)
    return fig