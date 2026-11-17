from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .normalize import normalize_text

GROUP_CANDIDATES = ("source", "episode", "cartoon", "show", "series", "video_source")


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"Text", "Class"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain Text and Class; missing {sorted(missing)}")
    df = df.reset_index(names="id")
    df["text"] = df["Text"].map(normalize_text)
    df["label"] = pd.to_numeric(df["Class"], errors="raise").astype(int)
    if not set(df.label.unique()).issubset({0, 1}):
        raise ValueError("Class must be binary values 0 and 1")
    return df


def find_group_column(columns: Iterable[str]) -> str | None:
    lookup = {str(c).lower().replace(" ", "_"): c for c in columns}
    return next((str(lookup[x]) for x in GROUP_CANDIDATES if x in lookup), None)


def make_splits(df: pd.DataFrame, seed: int = 42) -> tuple[dict[str, list[int]], str | None]:
    """70/15/15 split. Group split if an explicit source-like metadata column exists."""
    group_col = find_group_column(df.columns)
    ids = np.asarray(df.id)
    if group_col and df[group_col].notna().sum() and df[group_col].nunique() > 1:
        groups = df[group_col].fillna("__missing__")
        first = GroupShuffleSplit(n_splits=1, test_size=.30, random_state=seed)
        train_pos, temp_pos = next(first.split(ids, df.label, groups))
        second = GroupShuffleSplit(n_splits=1, test_size=.50, random_state=seed + 1)
        val_rel, test_rel = next(second.split(temp_pos, df.label.iloc[temp_pos], groups.iloc[temp_pos]))
        return {"train": ids[train_pos].tolist(), "val": ids[temp_pos[val_rel]].tolist(), "test": ids[temp_pos[test_rel]].tolist()}, group_col
    train_ids, temp_ids, train_y, temp_y = train_test_split(ids, df.label, test_size=.30, random_state=seed, stratify=df.label)
    val_ids, test_ids = train_test_split(temp_ids, test_size=.50, random_state=seed, stratify=temp_y)
    return {"train": train_ids.tolist(), "val": val_ids.tolist(), "test": test_ids.tolist()}, None


def save_prepared(df: pd.DataFrame, splits: dict[str, list[int]], out_dir: str | Path, group_col: str | None) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "prepared.csv", index=False)
    (out / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    (out / "split_metadata.json").write_text(json.dumps({"group_column": group_col}, indent=2), encoding="utf-8")


def split_frame(df: pd.DataFrame, splits: dict[str, list[int]], name: str) -> pd.DataFrame:
    return df.set_index("id").loc[splits[name]].reset_index()
