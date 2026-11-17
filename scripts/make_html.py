#!/usr/bin/env python3
"""Build one self-contained attribution viewer per model from JSONL rationale output."""
import argparse,html,json,random
from pathlib import Path

def main(args):
 random.seed(args.seed); root=Path(args.rationales); out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 for model in ("muril","xlmr","hybrid"):
  paths=list(root.glob(f"{model}_*.jsonl"));
  if not paths: continue
  rows=[]
  for line in paths[0].read_text(encoding="utf8").splitlines():
   x=json.loads(line)
   if x["prob"] >= .5: rows.append(x)
  chosen=random.sample(rows,min(50,len(rows)))+sorted(rows,key=lambda x:x["prob"],reverse=True)[:20]
  cards=[]
  for r in chosen:
   high={i for span in r["spans"] for i,w in enumerate(r["words"]) if w["char_start"]>=span["start"] and w["char_end"]<=span["end"]}
   words=" ".join(f'<mark class="hit" title="{w["score"]:.4f}">{html.escape(w["display"] or w["word"])}</mark>' if i in high else html.escape(w["word"]) for i,w in enumerate(r["words"]))
   cards.append(f'<article><small>id={r["id"]} gold={r["gold"]} p={r["prob"]:.3f}</small><p>{words}</p><code>{html.escape(" | ".join(s["text"] for s in r["spans"]))}</code></article>')
  (out/f"{model}.html").write_text(f'<!doctype html><meta charset="utf-8"><title>{model} rationales</title><style>body{{font:16px system-ui;max-width:900px;margin:auto}}article{{border-bottom:1px solid #ddd;padding:12px 0}}.hit{{background:#ffb4a2}}code{{color:#a23}}</style><h1>{model}: attribution-based rationale extraction</h1>'+"\n".join(cards),encoding="utf8")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--rationales",default="outputs/rationales");p.add_argument("--output",default="outputs/figures");p.add_argument("--seed",type=int,default=42);main(p.parse_args())
