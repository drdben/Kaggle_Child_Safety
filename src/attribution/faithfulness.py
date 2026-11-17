from __future__ import annotations
import random

def altered_text(words, selected, only=False, mask=None):
    chosen=set(selected)
    keep=[w["word"] if i in chosen else (mask if only and mask else "") for i,w in enumerate(words)] if only else [w["word"] if i not in chosen else (mask or "") for i,w in enumerate(words)]
    return " ".join(x for x in keep if x)

def summary(original_probs, deletion_probs, only_probs, original_pred, deletion_pred, words, selected):
    n=len(words) or 1
    return {"comprehensiveness":float((original_probs-deletion_probs).mean()),"sufficiency":float((original_probs-only_probs).mean()),"flip_rate":float(((original_pred==1)&(deletion_pred==0)).mean()),"mean_rationale_words":sum(len(x) for x in selected)/len(selected),"fraction_covered":sum(len(x)/n for x in selected)/len(selected)}
