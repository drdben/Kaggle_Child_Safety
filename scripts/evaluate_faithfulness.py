#!/usr/bin/env python3
"""Evaluate deletion, mask replacement, sufficiency, and equal-length random controls."""
import argparse,json,random,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.train import build
from src.attribution.faithfulness import altered_text

def main(a):
 cfg=yaml.safe_load(Path(a.config).read_text()); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model,toks=build(cfg); model.to(dev); model.load_state_dict(torch.load(a.checkpoint,map_location=dev,weights_only=False)["state_dict"]);model.eval()
 def prob(texts):
  out=[]
  for text in texts:
   e={n:t(text,return_tensors="pt",truncation=True,max_length=cfg["max_len"]).to(dev) for n,t in toks.items()}
   with torch.no_grad():
    z=model(e["muril"]["input_ids"],e["muril"]["attention_mask"],e["xlmr"]["input_ids"],e["xlmr"]["attention_mask"]) if cfg["model"]=="hybrid" else model(e[cfg["model"]]["input_ids"],e[cfg["model"]]["attention_mask"])
   out.append(torch.softmax(z,-1)[0,1].item())
  return np.array(out)
 results=[]
 for path in Path(a.rationales).glob(f"{cfg['model']}_seed*_.jsonl"):
  pass # retained for backwards-compatible glob handling below
 for path in Path(a.rationales).glob(f"{cfg['model']}_seed*_*.jsonl"):
  rows=[json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if json.loads(x)["pred"] == 1]
  if not rows: continue
  selected=[[i for i,w in enumerate(r["words"]) if any(w["char_start"]>=s["start"] and w["char_end"]<=s["end"] for s in r["spans"])] for r in rows]
  original=np.array([r["prob"] for r in rows]); deleted=prob([altered_text(r["words"],s) for r,s in zip(rows,selected)]); only=prob([altered_text(r["words"],s,only=True) for r,s in zip(rows,selected)])
  mask=next(iter(toks.values())).mask_token or "[MASK]"; masked=prob([altered_text(r["words"],s,mask=mask) for r,s in zip(rows,selected)])
  random_selected=[random.sample(range(len(r["words"])),min(len(s),len(r["words"]))) for r,s in zip(rows,selected)]
  rnd=prob([altered_text(r["words"],s) for r,s in zip(rows,random_selected)])
  coverage=[len(s)/max(1,len(row["words"])) for row,s in zip(rows,selected)]
  results.append({"file":path.name,"n":len(rows),"comprehensiveness":float((original-deleted).mean()),"sufficiency":float((original-only).mean()),"flip_rate":float((deleted<.5).mean()),"mask_comprehensiveness":float((original-masked).mean()),"random_comprehensiveness":float((original-rnd).mean()),"ig_random_gap":float((rnd-deleted).mean()),"mean_rationale_words":float(np.mean([len(x) for x in selected])),"fraction_covered":float(np.mean(coverage))})
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True); pd.DataFrame(results).to_csv(out/f"{cfg['model']}_faithfulness.csv",index=False)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--rationales",default="outputs/rationales");p.add_argument("--output",default="outputs/metrics");main(p.parse_args())
