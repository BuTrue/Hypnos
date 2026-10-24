# -*- coding: utf-8 -*-
"""Regenerate benchmark figures from results/benchmark_master.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / "results" / "benchmark_master.json").read_text(encoding="utf-8"))
rows = sorted(rows, key=lambda r: r["sum"])
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COL = {"Hypnos-V446 (legacy)": "#c0392b",
       "Hypnos-V448 (legacy)": "#e67e22",
       "Hypnos-v446fusion (CV5, OOF)": "#8e44ad"}

fig, ax = plt.subplots(figsize=(11, 5.2))
y = np.arange(len(rows))
h = 0.26
for i, (k, lab) in enumerate([("R", "REM"), ("W", "Wake"), ("N", "NREM")]):
    ax.barh(y + (i - 1) * h, [r[k] for r in rows], h, label=lab)
ax.set_yticks(y)
ax.set_yticklabels([r["name"] for r in rows], fontsize=8)
ax.set_xlim(0, 1.0)
ax.set_xlabel("F1")
ax.legend(ncol=3, loc="lower right")
ax.set_title("HYX 5-fold (5 s) - per-class F1")
fig.tight_layout()
fig.savefig(FIG / "f1_by_class.png", dpi=200)
plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(13, 8))
ax = axs[0, 0]
ax.barh(y, [r["sum"] for r in rows],
        color=[COL.get(r["name"], "#5d8aa8") for r in rows])
ax.set_yticks(y)
ax.set_yticklabels([r["name"] for r in rows], fontsize=7)
ax.set_xlabel("sum of three F1")
ax.set_title("Accuracy (sum F1)")
ax.set_xlim(1.0, max(r["sum"] for r in rows) + 0.05)

ax = axs[0, 1]
for r in rows:
    if r.get("infer_s") and r.get("size_mb"):
        ax.scatter(r["size_mb"], r["infer_s"], s=45,
                   color=COL.get(r["name"], "#5d8aa8"))
        ax.annotate(r["name"], (r["size_mb"], r["infer_s"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("model size (MB, log)")
ax.set_ylabel("inference on 24 h file (s, log)")
ax.set_title("Speed vs size")

ax = axs[1, 0]
tr = [(r["train_min_fold"], r["name"]) for r in rows if r.get("train_min_fold")]
tr.sort()
ax.barh([n for v, n in tr], [v for v, n in tr],
        color=[COL.get(n, "#5d8aa8") for v, n in tr])
ax.set_xlabel("training time per fold (min)")
ax.set_title("Training cost")
ax.tick_params(axis="y", labelsize=7)

ax = axs[1, 1]
mem = [(r["param_mem_mb"], r["name"]) for r in rows if r.get("param_mem_mb")]
mem.sort()
ax.barh([n for v, n in mem], [v for v, n in mem],
        color=[COL.get(n, "#5d8aa8") for v, n in mem])
ax.set_xlabel("parameter memory (MB, fp32)")
ax.set_title("Static model memory")
ax.tick_params(axis="y", labelsize=7)
fig.suptitle("Hypnos benchmark - HYX 31 recordings, mouse-disjoint 5-fold, 5 s")
fig.tight_layout()
fig.savefig(FIG / "benchmark_summary.png", dpi=200)
plt.close(fig)
print("figures written", FIG)
