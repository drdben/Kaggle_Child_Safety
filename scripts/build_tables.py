#!/usr/bin/env python3
"""Aggregate per-seed metrics and prediction disagreements into portable CSV/Markdown."""
import json
from pathlib import Path
import pandas as pd

metrics=Path("outputs/metrics"); metrics.mkdir(parents=True,exist_ok=True)
rows=[]
for p in metrics.glob("*_seed*.json"):
 x=json.loads(p.read_text()); rows.append({"model":p.name.split("_seed")[0],"seed":x["seed"],"accuracy":x["accuracy"],"macro_f1":x["macro_f1"],"roc_auc":x["roc_auc"]})
df=pd.DataFrame(rows)
if not df.empty:
 df.to_csv(metrics/"classification_results.csv",index=False)
 summary=df.groupby("model")[["accuracy","macro_f1","roc_auc"]].agg(["mean","std"]); (metrics/"classification_results.md").write_text(summary.to_markdown(),encoding="utf8")
# The seeded prediction files support paper-style error inspection (first common seed only).
for seed in (13,42,2024):
 a=metrics/f"muril_seed{seed}_test.csv"; b=metrics/f"xlmr_seed{seed}_test.csv"
 if a.exists() and b.exists():
  left=pd.read_csv(a).set_index("id"); right=pd.read_csv(b).set_index("id"); j=left.join(right,lsuffix="_muril",rsuffix="_xlmr")
  d=j[((j.gold_muril==j.pred_muril)&(j.gold_muril!=j.pred_xlmr))|((j.gold_muril!=j.pred_muril)&(j.gold_muril==j.pred_xlmr))]
  d.to_csv(metrics/f"muril_xlmr_disagreements_seed{seed}.csv")
