# -*- coding: utf-8 -*-
"""Aggregate per-fold benchmark JSONs into the published row format.

Usage:
    python aggregate_external.py results_dir [--order R W N]

Each JSON is expected to contain ``fold0`` ... ``fold4`` with
``test_per_rec`` -> recording -> ``{f1_rem, f1_wake, f1_nrem}`` (or a 3-list in
the given class order).  Per fold, class F1 is averaged over recordings; the
final row is the mean over folds.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _vals(meta, order):
    if isinstance(meta, dict):
        return [meta.get("f1_rem"), meta.get("f1_wake"), meta.get("f1_nrem")]
    return list(meta)


def aggregate(path: Path, order=(0, 1, 2)) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    per = {"R": [], "W": [], "N": []}
    train = 0.0
    params = []
    for fold in sorted(k for k in d if k.startswith("fold")):
        recs = d[fold].get("test_per_rec", {})
        vals = [_vals(v, order) for v in recs.values()]
        if not vals:
            continue
        per["R"].append(statistics.mean(v[order[0]] for v in vals))
        per["W"].append(statistics.mean(v[order[1]] for v in vals))
        per["N"].append(statistics.mean(v[order[2]] for v in vals))
        train += float(d[fold].get("train_s", 0) or 0)
        train += float(d[fold].get("feat_s", 0) or 0)
        if d[fold].get("n_params"):
            params.append(int(d[fold]["n_params"]))
    if not per["R"]:
        return {}
    r, w, n = (statistics.mean(per[k]) for k in ("R", "W", "N"))
    return {
        "model": path.stem,
        "n_folds": len(per["R"]),
        "R": round(r, 4), "W": round(w, 4), "N": round(n, 4),
        "mean": round((r + w + n) / 3, 4), "sum": round(r + w + n, 4),
        "std_R": round(statistics.pstdev(per["R"]), 4),
        "params": statistics.median(params) if params else None,
        "train_min_fold": round(train / len(per["R"]) / 60, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--order", nargs=3, type=int, default=[0, 1, 2],
                    help="class order in each per-recording list")
    args = ap.parse_args()
    out = []
    for p in sorted(Path(args.results_dir).glob("*.json")):
        row = aggregate(p, tuple(args.order))
        if row:
            out.append(row)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
