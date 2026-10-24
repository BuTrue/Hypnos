# -*- coding: utf-8 -*-
"""Decoding and scoring.

Predictions are made on 5 s epochs but the native DB3 annotations are 10 s.
Scoring folds each pair of 5 s probabilities into one 10 s decision.  The
deployed decoder applies a 4.5 % REM prior (a global additive bias on the REM
log-probability) followed by Viterbi decoding with a transition matrix fitted
on the training recordings (lambda = 0.5).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

CLASS_NAMES = ("REM", "Wake", "NREM")
REM_PRIOR = 0.045
VITERBI_LAMBDA = 0.5


def fold_to_10s(probs: np.ndarray, y: np.ndarray, epoch_sec: float = 5.0):
    """Average sub-10 s probabilities inside each native annotation block."""
    factor = int(round(10.0 / float(epoch_sec)))
    if factor <= 1:
        return probs, y
    n = (len(y) // factor) * factor
    p = probs[:n].reshape(-1, factor, probs.shape[1]).mean(axis=1)
    blocks = y[:n].reshape(-1, factor)
    y10 = blocks[:, 0].copy()
    for i in range(len(y10)):
        vals, cnt = np.unique(blocks[i], return_counts=True)
        y10[i] = vals[np.argmax(cnt)]
    return p, y10


def fit_transition(y: np.ndarray, n_states: int = 3, eps: float = 1e-3) -> np.ndarray:
    counts = np.full((n_states, n_states), eps, dtype=np.float64)
    y = np.asarray(y)
    for a, b in zip(y[:-1], y[1:]):
        if a >= 0 and b >= 0:
            counts[a, b] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def bias_for(probs: np.ndarray, prior: float = REM_PRIOR) -> float:
    """Additive REM bias that makes the predicted REM rate equal ``prior``."""
    lo, hi = -8.0, 8.0
    for _ in range(30):
        b = 0.5 * (lo + hi)
        frac = float((probs[:, 0] + b > np.maximum(probs[:, 1], probs[:, 2])).mean())
        if frac < prior:
            lo = b
        else:
            hi = b
    return 0.5 * (lo + hi)


def viterbi(log_probs: np.ndarray, trans: np.ndarray, lam: float = VITERBI_LAMBDA) -> np.ndarray:
    t_len, k = log_probs.shape
    log_t = np.log(np.clip(trans, 1e-12, 1.0))
    dp = np.empty((t_len, k), dtype=np.float64)
    back = np.zeros((t_len, k), dtype=np.int16)
    dp[0] = log_probs[0]
    for t in range(1, t_len):
        cand = dp[t - 1][:, None] + lam * log_t
        back[t] = np.argmax(cand, axis=0)
        dp[t] = log_probs[t] + cand[back[t], np.arange(k)]
    path = np.empty(t_len, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(probs: np.ndarray, trans: np.ndarray,
           prior: float = REM_PRIOR, lam: float = VITERBI_LAMBDA) -> np.ndarray:
    """REM prior + Viterbi; call after ``fold_to_10s`` for the native protocol."""
    logp = np.log(np.clip(probs, 1e-9, 1.0))
    logp[:, 0] += bias_for(probs, prior)
    return viterbi(logp, trans, lam=lam)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    v = y_true >= 0
    y_true, y_pred = y_true[v], y_pred[v]
    f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "f1": {CLASS_NAMES[i]: round(float(f1[i]), 4) for i in range(3)},
        "mean_f1": round(float(np.mean(f1)), 4),
        "total_f1": round(float(np.sum(f1)), 4),
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
    }
