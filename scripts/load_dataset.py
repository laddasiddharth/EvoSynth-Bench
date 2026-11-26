"""
EvoSynth-Bench v2 loader and validator (standalone: needs numpy, pandas, scikit-learn;
torchvision only for Fashion-MNIST).

The datasets themselves are NOT redistributed here. They are fetched from their original
sources (scikit-learn / OpenML / torchvision), preprocessed exactly as EvoSynth does, and split
with the row-index files in data/splits/.

    python scripts/load_dataset.py                 # validate the split files (no download)
    python scripts/load_dataset.py --fetch adult   # download one dataset and check it

    from load_dataset import load_splits
    parts = load_splits("adult")                   # {"test": (X, y), "search": ..., "train": ...}

Preprocessing (identical to EvoSynth's src/evosynth/datasets/loaders.py):
  - labels re-coded to 0..K-1 (sorted original labels)
  - categorical / text columns -> integer codes (categories sorted), missing -> its own code
  - missing numeric values -> the column median
"""
import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def catalogue():
    with open(os.path.join(DATA, "datasets.csv"), encoding="utf-8") as f:
        return {row["dataset"]: row for row in csv.DictReader(f)}


def split_indices(name):
    """{split name: sorted row indices} from data/splits/<name>.csv."""
    out = {}
    with open(os.path.join(DATA, "splits", f"{name}.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for part in row["split"].split("+"):
                out.setdefault(part, []).append(int(row["row_index"]))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def _frame_to_arrays(frame, target):
    import pandas as pd
    _, y = np.unique(frame[target].astype(str).to_numpy(dtype=object), return_inverse=True)
    cols = []
    for col in frame.columns:
        if col == target:
            continue
        s = frame[col]
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_string_dtype(s):
            cats = sorted(s.dropna().astype(str).unique())
            lookup = {c: i for i, c in enumerate(cats)}
            codes = s.astype(object).map(lambda v: lookup.get(str(v), len(cats)) if pd.notna(v) else len(cats))
            cols.append(codes.to_numpy(dtype=np.float32))
        else:
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
            if np.isnan(v).any():
                v = np.where(np.isnan(v), np.nanmedian(v), v)
            cols.append(v.astype(np.float32))
    return np.column_stack(cols), y.astype(np.int64)


def load_full(name, data_home=None):
    """The whole dataset as (X, y) in source row order (Fashion-MNIST: the 60,000 training images)."""
    info = catalogue()[name]
    if info["source"] == "sklearn.fetch_covtype":
        from sklearn.datasets import fetch_covtype
        d = fetch_covtype(data_home=data_home)
        _, y = np.unique(d.target, return_inverse=True)
        return d.data.astype(np.float32), y.astype(np.int64)
    if info["source"] == "openml":
        from sklearn.datasets import fetch_openml
        b = fetch_openml(data_id=int(info["openml_id"]), data_home=data_home, as_frame=True, parser="auto")
        target = b.target.name if hasattr(b.target, "name") else b.target_names[0]
        frame = b.frame if b.frame is not None else b.data.assign(**{target: b.target})
        return _frame_to_arrays(frame, target)
    if info["source"] == "torchvision.FashionMNIST":
        from torchvision.datasets import FashionMNIST
        ds = FashionMNIST(root=data_home or os.path.join(ROOT, ".data"), train=True, download=True)
        return ds.data.numpy().astype(np.float32)[:, None] / 255.0, ds.targets.numpy().astype(np.int64)
    raise ValueError(f"unknown source {info['source']}")


def load_splits(name, data_home=None):
    """{split: (X, y)} exactly as EvoSynth used them (data_seed 42)."""
    X, y = load_full(name, data_home)
    return {k: (X[idx], y[idx]) for k, idx in split_indices(name).items()}


def validate():
    """Check every split file against the catalogue (sizes, no overlap with test). No download."""
    ok = True
    for name, info in catalogue().items():
        sp = split_indices(name)
        if info["kind"] == "tabular":
            checks = [len(sp["test"]) == int(info["test"]), len(sp["search"]) == int(info["search"]),
                      len(sp["train"]) == int(info["train"]),
                      not set(sp["test"]) & (set(sp["search"]) | set(sp["train"])),
                      max(max(v) for v in sp.values()) < int(info["rows"])]
        else:
            checks = [len(sp["search_train"]) == int(info["train"]),
                      len(sp["search_train"]) + len(sp["search_val"]) == int(info["search"]),
                      not set(sp["search_train"]) & set(sp["search_val"])]
        status = "OK  " if all(checks) else "FAIL"
        ok &= all(checks)
        print(f"  [{status}] {name:<14} " + ", ".join(f"{k} {len(v):,}" for k, v in sp.items()))
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--fetch", metavar="NAME", help="download one dataset and check it against the catalogue")
    a = p.parse_args()
    print("EvoSynth-Bench v2: split files")
    good = validate()
    if a.fetch:
        info = catalogue()[a.fetch]
        parts = load_splits(a.fetch)
        X = parts[next(iter(parts))][0]
        print(f"  [{a.fetch}] " + ", ".join(f"{k}: {v[0].shape}" for k, v in parts.items()))
        good &= info["kind"] == "image" or X.shape[1] == int(info["features"])
    print("Validation PASSED." if good else "Validation FAILED.")
    sys.exit(0 if good else 1)
