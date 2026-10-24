# -*- coding: utf-8 -*-
"""Minimal 5-fold training script for v446fusion.

Expected data: one ``.npz`` per recording with
``img`` float32 (184, T) and ``cls`` int64 (T,), labels 0 REM / 1 Wake /
2 NREM / -1 invalid.  ``--folds`` is a JSON list of
``{"train": [rec...], "val": [rec...], "test": [rec...]}`` objects.

Example
-------
    python -m hypnos.train --data cache --folds folds.json \
        --out runs --epochs 10 --epi 22
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import decode, features
from .models import V446Fusion


@dataclass
class Recording:
    rec: str
    q: np.ndarray
    y: np.ndarray


def load_recording(path: Path, epi: int) -> Recording:
    img, cls = features.load_cache(path)
    q = features.prepare_recording(img, cls, epi=epi)
    return Recording(path.stem, q, cls)


class WindowDataset(Dataset):
    def __init__(self, recs: list[Recording], epi: int, drop_boundary: int = 0):
        self.recs, self.epi = recs, epi
        idx = []
        for ri, r in enumerate(recs):
            valid = np.flatnonzero(r.y >= 0)
            if drop_boundary > 0:
                chg = np.zeros(len(r.y), dtype=np.int64)
                chg[1:] = (r.y[1:] != r.y[:-1]).astype(np.int64)
                near = np.convolve(chg, np.ones(2 * drop_boundary + 1,
                                                dtype=np.int64), mode="same") > 0
                valid = valid[~near[valid]]
            idx.extend((ri, int(j)) for j in valid)
        self.index = idx

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ri, j = self.index[i]
        r = self.recs[ri]
        x = r.q[:, j:j + self.epi].astype(np.float32)[None]
        return torch.from_numpy(x), int(r.y[j])


def predict(model, recs: list[Recording], epi: int, batch: int = 256) -> dict[str, np.ndarray]:
    model.eval()
    out = {}
    device = next(model.parameters()).device
    with torch.no_grad():
        for r in recs:
            n = len(r.y)
            probs = np.empty((n, 3), dtype=np.float32)
            for lo in range(0, n, batch):
                hi = min(lo + batch, n)
                x = np.stack([r.q[:, j:j + epi] for j in range(lo, hi)])
                x = torch.from_numpy(x[:, None].astype(np.float32)).to(device)
                probs[lo:hi] = torch.softmax(model(x), 1).cpu().numpy()
            out[r.rec] = probs
    return out


def score(recs: list[Recording], probs: dict[str, np.ndarray],
          trans: np.ndarray) -> dict:
    per = {}
    for r in recs:
        p10, y10 = decode.fold_to_10s(probs[r.rec], r.y, 5.0)
        pred = decode.decode(p10, trans)
        per[r.rec] = decode.metrics(y10, pred)
    return {
        "per_rec": per,
        "rem": float(np.mean([v["f1"]["REM"] for v in per.values()])),
        "mean": float(np.mean([v["mean_f1"] for v in per.values()])),
        "total": float(np.mean([v["total_f1"] for v in per.values()])),
    }


def run_fold(fold: dict, data_dir: Path, epi: int, args, fi: int) -> dict:
    train = [load_recording(data_dir / f"{r}.npz", epi) for r in fold["train"]]
    val = [load_recording(data_dir / f"{r}.npz", epi) for r in fold["val"]]
    test = [load_recording(data_dir / f"{r}.npz", epi) for r in fold["test"]]
    ds = WindowDataset(train, epi, drop_boundary=args.drop_boundary)
    labels = np.array([train[ri].y[j] for ri, j in ds.index])
    counts = np.bincount(labels, minlength=3)
    cw = torch.tensor(len(labels) / (3.0 * np.maximum(counts, 1)), dtype=torch.float32)
    trans = decode.fit_transition(np.concatenate([r.y for r in train]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + fi)
    model = V446Fusion(epi=epi).to(device)
    crit = nn.CrossEntropyLoss(weight=cw.to(device))
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.wd)
    steps = int(np.ceil(len(ds) / args.batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps * args.epochs)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    best = (-1.0, None, 0)
    for ep in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        val_scores = score(val, predict(model, val, epi), trans)
        print(f"[fold{fi}] ep{ep + 1} val REM={val_scores['rem']:.4f} "
              f"mean={val_scores['mean']:.4f}", flush=True)
        if val_scores["rem"] > best[0]:
            best = (val_scores["rem"], {k: v.clone() for k, v in model.state_dict().items()}, ep + 1)

    model.load_state_dict(best[1])
    test_scores = score(test, predict(model, test, epi), trans)
    return {"fold": fi, "best_epoch": best[2], "val_rem": best[0],
            "test": test_scores, "params": sum(p.numel() for p in model.parameters())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory of <rec>.npz")
    ap.add_argument("--folds", required=True, help="JSON list of train/val/test splits")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--epi", type=int, default=22)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--drop-boundary", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    folds = json.loads(Path(args.folds).read_text(encoding="utf-8"))
    data_dir = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for fi, fold in enumerate(folds):
        res = run_fold(fold, data_dir, args.epi, args, fi)
        results.append(res)
        (out / f"fold{fi}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    summary = {
        "folds": results,
        "oof_rem": float(np.mean([r["test"]["rem"] for r in results])),
        "oof_mean": float(np.mean([r["test"]["mean"] for r in results])),
        "oof_total": float(np.mean([r["test"]["total"] for r in results])),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
