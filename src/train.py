from __future__ import annotations
import argparse, json, random, subprocess
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from .data import load_dataset, split_frame
from .evaluate import classification_metrics, save_metrics
from .models.classifier import EncoderClassifier
from .models.hybrid import HybridClassifier


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class TextDataset(Dataset):
    def __init__(self, frame, tokenizers, max_len): self.frame, self.tokenizers, self.max_len = frame, tokenizers, max_len
    def __len__(self): return len(self.frame)
    def __getitem__(self, i):
        row = self.frame.iloc[i]; out = {"label": torch.tensor(row.label), "id": torch.tensor(row.id)}
        for name, tok in self.tokenizers.items():
            x = tok(row.text, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
            out.update({f"{name}_{k}": v.squeeze(0) for k,v in x.items()})
        return out


def build(config):
    names = config["model_names"]
    if config["model"] == "hybrid":
        toks = {"muril": AutoTokenizer.from_pretrained(names["muril"], use_fast=True), "xlmr": AutoTokenizer.from_pretrained(names["xlmr"], use_fast=True)}
        return HybridClassifier(names["muril"], names["xlmr"], config["dropout"]), toks
    name = names[config["model"]]
    return EncoderClassifier(name, config["dropout"]), {config["model"]: AutoTokenizer.from_pretrained(name, use_fast=True)}


def logits(model, batch, model_name):
    if model_name == "hybrid": return model(batch["muril_input_ids"], batch["muril_attention_mask"], batch["xlmr_input_ids"], batch["xlmr_attention_mask"])
    return model(batch[f"{model_name}_input_ids"], batch[f"{model_name}_attention_mask"])


@torch.no_grad()
def predict(model, loader, device, model_name):
    model.eval(); ys=[]; ps=[]; ids=[]
    for b in loader:
        b={k:v.to(device) for k,v in b.items()}; ps.append(torch.softmax(logits(model,b,model_name),-1).cpu()); ys.append(b["label"].cpu()); ids.append(b["id"].cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(ids).numpy()


def git_hash():
    try: return subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception: return "unknown"


def run(config):
    out=Path(config["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    df=load_dataset(config["data_csv"]); splits=json.loads(Path(config["splits_path"]).read_text())
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_rows=[]
    for seed in config["seeds"]:
        seed_everything(seed); model,toks=build(config); model.to(device)
        loaders={n:DataLoader(TextDataset(split_frame(df,splits,n),toks,config["max_len"]), batch_size=config["batch_size"], shuffle=n=="train") for n in splits}
        counts=split_frame(df,splits,"train").label.value_counts()
        # Python division produces float64 values; CrossEntropyLoss requires weights
        # to match the float32 logits unless mixed precision has explicitly changed them.
        weights=torch.tensor(
            [len(split_frame(df,splits,"train")) / (2 * counts.get(i, 1)) for i in range(2)],
            dtype=torch.float32,
            device=device,
        )
        opt=AdamW(model.parameters(),lr=float(config["lr"]),weight_decay=float(config["weight_decay"])); total=len(loaders["train"])*config["epochs"]
        sched=get_linear_schedule_with_warmup(opt,int(total*config["warmup_ratio"]),total); loss_fn=CrossEntropyLoss(weight=weights)
        best=-1; stale=0; checkpoint=out/f"{config['model']}_seed{seed}.pt"
        for epoch in range(config["epochs"]):
            model.train()
            for b in tqdm(loaders["train"],desc=f"seed {seed} epoch {epoch+1}"):
                b={k:v.to(device) for k,v in b.items()}; opt.zero_grad(); loss=loss_fn(logits(model,b,config["model"]),b["label"]); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),config["grad_clip"]); opt.step(); sched.step()
            vy,vp,_=predict(model,loaders["val"],device,config["model"]); score=classification_metrics(vy,vp)["macro_f1"]
            if score > best: best=score; stale=0; torch.save({"state_dict":model.state_dict(),"config":config,"seed":seed},checkpoint)
            else: stale+=1
            if stale >= config["patience"]: break
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)["state_dict"]); ty,tp,ids=predict(model,loaders["test"],device,config["model"]); met=classification_metrics(ty,tp); met.update({"seed":seed,"git_hash":git_hash()}); save_metrics(met,out/f"{config['model']}_seed{seed}.json")
        pd.DataFrame({"id":ids,"gold":ty,"pred":tp.argmax(1),"prob_profane":tp[:,1]}).to_csv(out/f"{config['model']}_seed{seed}_test.csv",index=False); run_rows.append({"seed":seed,**{k:met[k] for k in ("accuracy","macro_f1","roc_auc")}})
    summary=pd.DataFrame(run_rows); summary.to_csv(out/f"{config['model']}_summary.csv",index=False); print(summary.agg(["mean","std"]))

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); run(yaml.safe_load(Path(a.config).read_text()))
