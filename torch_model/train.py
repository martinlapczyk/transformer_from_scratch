import torch
import torch.nn as nn
from dataset import load_multi30k,TranslationDataset,collate_batch, PAD
from model import Transformer
device="cuda" if torch.cuda.is_available() else "cpu"
train_src, train_tgt=load_multi30k(split="train")
train_dataset=TranslationDataset(train_src,train_tgt)
pad_idx=train_dataset.src_vocab.stoi[PAD]

loader=torch.utils.data.DataLoader(
    train_dataset,batch_size=32,shuffle=True,
    collate_fn=lambda b:collate_batch(b,pad_idx=pad_idx)
)

model=Transformer(
    src_vocab_size=len(train_dataset.src_vocab),
    tgt_vocab_size=len(train_dataset.tgt_vocab),
    pad_idx=pad_idx,
).to(device)

optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
def train_step(model,src,tgt,optimizer,pad_idx=0):
    tgt_input=tgt[:,:-1]
    tgt_output=tgt[:,1:]

    logits,_=model(src,tgt_input)
    loss=torch.nn.functional.cross_entropy(
        logits.reshape(-1,logits.size(-1)),tgt_output.reshape(-1),ignore_index=pad_idx
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

if __name__=="__main__":
    epochs=10
    for epoch in range(epochs):
        total_loss=0
        for src,tgt in loader:
            src,tgt=src.to(device),tgt.to(device)
            loss=train_step(model,src,tgt,optimizer,pad_idx=pad_idx)
            total_loss+=loss
        print(f"epoch {epoch+1},loss {total_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), "../checkpoints/model.pt")