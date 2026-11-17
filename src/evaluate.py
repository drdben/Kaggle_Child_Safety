from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score


def classification_metrics(y_true, probabilities) -> dict:
    y_true, p = np.asarray(y_true), np.asarray(probabilities)
    pred = p.argmax(1)
    return {"accuracy": float(accuracy_score(y_true, pred)), "macro_f1": float(f1_score(y_true, pred, average="macro")),
            "roc_auc": float(roc_auc_score(y_true, p[:, 1])) if len(np.unique(y_true)) == 2 else None,
            "per_class": classification_report(y_true, pred, labels=[0, 1], output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist()}


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
