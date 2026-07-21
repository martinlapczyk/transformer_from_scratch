@torch.no_grad()

def translate_sentence(model,src_sentence_ids,src_vocab,tgt_vocab,max_len=50,device="cpu"):
    model.eval()
    src=torch.tensor(src_sentence_ids,dtype=torch.long,device=device).unsqueeze(0)
    src_mask=model.make_src_mask(src)
    encoder_out=model.encoder(src,src_mask)

    tgt_ids=[tgt_vocab.stoi[SOS]]
    for _ in range(max_len):
        tgt.torch.tensor(tgt_ids,dtype=torch.long,device=device).unsqueeze(0)
        causal_mask=model.make_causal_mask(tgt)
        decoder_out,cross_weights=model.decoder(tgt,encoder_out,causal_mask,src_mask)
        logits=model.out_proj(decoder_out)
        next_id=logits[0,-1,:].argmax().item()
        tgt_ids.append(next_id)
        if next_id ==tgt_vocab.stoi[EOS]:
            break
    
    output_ids=tgt_ids[1:]
    if output_ids and output_ids[-1]==tgt_vocab.stoi[EOS]:
        output_ids=output_ids[:-1]
    return tgt_vocab.decode(output_ids), cross_weights
