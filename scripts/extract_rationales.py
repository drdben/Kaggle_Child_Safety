#!/usr/bin/env python3
"""Generate per-strategy JSONL rationales for a trained checkpoint's test predictions."""
import argparse,json,sys
from pathlib import Path
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import load_dataset,split_frame
from src.train import build,logits
from src.attribution.ig import single_ig,hybrid_ig
from src.attribution.align import subtoken_to_words,normalize_word_scores
from src.attribution.spans import select_words,merge_spans

def main(c):
 cfg=yaml.safe_load(Path(c.config).read_text()); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model,toks=build(cfg); model.to(device)
 ck=torch.load(c.checkpoint,map_location=device,weights_only=False); model.load_state_dict(ck["state_dict"]); model.eval()
 splits=json.loads(Path(cfg["splits_path"]).read_text()); frame=split_frame(load_dataset(cfg["data_csv"]),splits,c.split)
 outputs={f"relative_{x}":[] for x in (.1,.2,.3,.5)}|{f"top_percent_{x}":[] for x in (.1,.2,.3)}|{"top1":[]}
 for _,r in frame.iterrows():
  enc={n:t(r.text,return_offsets_mapping=True,return_tensors="pt",truncation=True,max_length=cfg["max_len"]) for n,t in toks.items()}
  enc={n:{k:v.to(device) for k,v in x.items()} for n,x in enc.items()}
  with torch.no_grad():
   if cfg["model"]=="hybrid": out=logits(model,{f"{n}_{k}":v for n,x in enc.items() for k,v in x.items() if k!="offset_mapping"},"hybrid")
   else: n=cfg["model"]; out=model(enc[n]["input_ids"],enc[n]["attention_mask"])
   prob=torch.softmax(out,-1)[0,1].item(); pred=int(out.argmax(-1).item())
  # Retain the predicted-positive evaluation set and gold-positive audit set (their union).
  if pred != 1 and int(r.label) != 1: continue
  if cfg["model"]=="hybrid":
   ma,xa,delta=hybrid_ig(model,enc["muril"]["input_ids"],enc["muril"]["attention_mask"],enc["xlmr"]["input_ids"],enc["xlmr"]["attention_mask"],toks["muril"],toks["xlmr"],cfg["ig_steps"])
   mw=normalize_word_scores(subtoken_to_words(r.text,enc["muril"]["offset_mapping"][0].cpu(),ma[0],cfg["word_aggregation"]),cfg["normalization"]); xw=normalize_word_scores(subtoken_to_words(r.text,enc["xlmr"]["offset_mapping"][0].cpu(),xa[0],cfg["word_aggregation"]),cfg["normalization"])
   words=[{**a,"score":cfg["hybrid_lambda"]*a["score"]+(1-cfg["hybrid_lambda"])*b["score"],"muril_score":a["score"],"xlmr_score":b["score"]} for a,b in zip(mw,xw)]
  else:
   n=cfg["model"]; a,delta=single_ig(model,enc[n]["input_ids"],enc[n]["attention_mask"],toks[n],cfg["ig_steps"]); words=normalize_word_scores(subtoken_to_words(r.text,enc[n]["offset_mapping"][0].cpu(),a[0],cfg["word_aggregation"]),cfg["normalization"])
  for name,rows in outputs.items():
   strategy,value=("top1",1) if name=="top1" else name.rsplit("_",1); chosen=select_words(words,strategy,float(value)); rows.append({"id":int(r.id),"text":r.text,"gold":int(r.label),"pred":pred,"prob":prob,"model":cfg["model"],"ig_delta":float(delta.flatten()[0]),"words":words,"spans":merge_spans(r.text,words,chosen)})
 out=Path(c.output);out.mkdir(parents=True,exist_ok=True)
 seed=ck.get("seed","unknown")
 for strategy,rows in outputs.items(): (out/f"{cfg['model']}_seed{seed}_{strategy}.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--output",default="outputs/rationales");p.add_argument("--split",default="test",choices=["test","val"]);main(p.parse_args())
