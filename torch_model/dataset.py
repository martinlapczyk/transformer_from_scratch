import torch
from torch.utils.data import Dataset
from collections import Counter
from datasets import load_dataset

PAD,SOS,EOS,UNK="<pad>","<sos>","<eos>","<unk>"
SPECIAL_TOKENS=[PAD,SOS,EOS,UNK]

class Vocab:
    def __init__(self, token_lists,min_freq=2):
        counter=Counter(tok for tokens in token_lists for tok in tokens)
        self.itos=list(SPECIAL_TOKENS)+[
            tok for tok, freq in counter.items() if freq >=min_freq
        ]
        self.stoi={tok: i for i, tok in enumerate(self.itos)}
    
    def __len__(self):
        return len(self.itos)
    def encode(self,tokens):
        unk_id=self.stoi[UNK]
        return [self.stoi.get(tok,unk_id)for tok in tokens]
    def decode(self,ids):
        return [self.itos[i] for i in ids]

def simple_tokenize(text):
    text=text.lower().strip()
    for ch in ".,!?;:":
        text=text.replace(ch,f" {ch} ")
    return text.split()

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab=None, tgt_vocab=None, min_freq=2):
        assert len(src_sentences)==len(tgt_sentences)
        self.src_tokens=[simple_tokenize(s) for s in src_sentences]
        self.tgt_tokens=[simple_tokenize(s) for s in tgt_sentences]
        self.src_vocab=src_vocab or Vocab(self.src_tokens,min_freq=min_freq)
        self.tgt_vocab=tgt_vocab or Vocab(self.tgt_tokens,min_freq=min_freq)
    
    def __len__(self):
        return len(self.src_tokens)
    
    def __getitem__(self,idx):
        src_ids=self.src_vocab.encode(self.src_tokens[idx])
        tgt_ids=[self.tgt_vocab.stoi[SOS]]+self.tgt_vocab.encode(self.tgt_tokens[idx])+[self.tgt_vocab.stoi[EOS]]
        return torch.tensor(src_ids,dtype=torch.long),torch.tensor(tgt_ids,dtype=torch.long)

def collate_batch(batch,pad_idx=0):
    src_batch,tgt_batch=zip(*batch)
    src_padded=torch.nn.utils.rnn.pad_sequence(src_batch,batch_first=True,padding_value=pad_idx)
    tgt_padded=torch.nn.utils.rnn.pad_sequence(tgt_batch,batch_first=True,padding_value=pad_idx)
    return src_padded,tgt_padded
    
def load_multi30k(split="train"):
    ds=load_dataset("bentrevett/multi30k",split=split)
    src_sentences=[ex["en"] for ex in ds]
    tgt_sentences=[ex["de"] for ex in ds]
    return src_sentences, tgt_sentences