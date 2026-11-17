#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_dataset, make_splits, save_prepared, split_frame

p=argparse.ArgumentParser(); p.add_argument("--input",default="AnnotationFinal.csv"); p.add_argument("--output",default="data"); p.add_argument("--seed",type=int,default=42); args=p.parse_args()
df=load_dataset(args.input); splits,group=make_splits(df,args.seed); save_prepared(df,splits,args.output,group)
print("Class distribution:",df.label.value_counts().sort_index().to_dict()); print("Split method:",f"grouped by {group}" if group else "stratified random")
for n in splits: print(n,len(splits[n]),split_frame(df,splits,n).label.value_counts().sort_index().to_dict())
