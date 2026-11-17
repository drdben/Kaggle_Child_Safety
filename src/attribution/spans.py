from __future__ import annotations
import math

def select_words(words, strategy: str, value: float):
    positive=[(i,w) for i,w in enumerate(words) if w["score"] > 0]
    if not positive: return []
    if strategy == "relative":
        threshold=value*max(w["score"] for _,w in positive); return [i for i,w in positive if w["score"] > threshold]
    if strategy == "top_percent":
        count=max(1,math.ceil(len(words)*value)); return [i for i,_ in sorted(positive,key=lambda x:x[1]["score"],reverse=True)[:count]]
    if strategy == "top1": return [max(positive,key=lambda x:x[1]["score"])[0]]
    raise ValueError(strategy)

def merge_spans(text, words, selected):
    selected=sorted(selected)
    groups=[]
    for i in selected:
        if not groups or i != groups[-1][-1]+1: groups.append([i])
        else: groups[-1].append(i)
    return [{"text":text[words[g[0]]["char_start"]:words[g[-1]]["char_end"]],"start":words[g[0]]["char_start"],"end":words[g[-1]]["char_end"],"score":sum(words[i]["score"] for i in g)} for g in groups]
