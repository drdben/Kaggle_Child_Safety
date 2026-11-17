from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass
class Word:
    word: str; char_start: int; char_end: int; score: float = 0.0

def whitespace_words(text: str) -> list[Word]:
    return [Word(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]

def subtoken_to_words(text: str, offsets, scores, aggregation: str = "sum") -> list[dict]:
    """Map fast-tokenizer character offsets to whitespace words; special (0,0) offsets are skipped."""
    words=whitespace_words(text); buckets=[[] for _ in words]
    for offset, score in zip(offsets, scores):
        start,end=map(int,offset)
        if end <= start: continue
        for i,w in enumerate(words):
            if start >= w.char_start and end <= w.char_end:
                buckets[i].append(float(score)); break
    out=[]
    for word, values in zip(words,buckets):
        raw = 0.0 if not values else {"sum":sum(values), "mean":sum(values)/len(values), "max":max(values)}[aggregation]
        out.append({"word":word.word, "display":word.word.strip("\"'.,!?;:()[]{}।"), "char_start":word.char_start, "char_end":word.char_end, "score":raw})
    return out

def normalize_word_scores(words: list[dict], mode: str = "l1") -> list[dict]:
    denom = sum(abs(x["score"]) for x in words) if mode == "l1" else sum(x["score"]**2 for x in words)**.5
    return [{**x, "score": x["score"] / denom if denom else 0.0} for x in words]
