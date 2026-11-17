"""Captum LayerIntegratedGradients for sentence-level classifier rationales."""
from __future__ import annotations
import torch
from captum.attr import LayerIntegratedGradients

def pad_baseline(ids, tokenizer):
    base=torch.full_like(ids, tokenizer.pad_token_id)
    # Preserve tokenizer-specific leading/trailing special tokens, while pad-baselining lexical tokens.
    special=set(tokenizer.all_special_ids)
    for b in range(ids.size(0)):
        for j, token in enumerate(ids[b].tolist()):
            if token in special: base[b,j]=token
    return base

def single_ig(model, ids, mask, tokenizer, n_steps=50):
    model.eval(); layer=model.encoder.get_input_embeddings()
    lig=LayerIntegratedGradients(lambda x,m: model(x,m),layer)
    attr,delta=lig.attribute(ids,baselines=pad_baseline(ids,tokenizer),additional_forward_args=(mask,),target=1,n_steps=n_steps,return_convergence_delta=True)
    return attr.sum(-1).detach().cpu(), delta.detach().cpu()

def hybrid_ig(model, muril_ids, muril_mask, xlmr_ids, xlmr_mask, muril_tok, xlmr_tok, n_steps=50):
    """Joint IG through the full hybrid model; returns one signed attribution vector per encoder."""
    model.eval(); layers=[model.muril.get_input_embeddings(),model.xlmr.get_input_embeddings()]
    lig=LayerIntegratedGradients(lambda a,b,am,bm: model(a,am,b,bm),layers)
    attrs,delta=lig.attribute((muril_ids,xlmr_ids),baselines=(pad_baseline(muril_ids,muril_tok),pad_baseline(xlmr_ids,xlmr_tok)),additional_forward_args=(muril_mask,xlmr_mask),target=1,n_steps=n_steps,return_convergence_delta=True)
    return attrs[0].sum(-1).detach().cpu(),attrs[1].sum(-1).detach().cpu(),delta.detach().cpu()
