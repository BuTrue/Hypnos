# -*- coding: utf-8 -*-
"""ONNX inference for the deployed Hypnos model.

Input to the network is float32 ``(B, 1, 184, context)`` with the training
scale 0..255 (the uint8 feature matrix cast to float).  Output is logits in
REM / Wake / NREM order.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import features


class HypnosONNX:
    def __init__(self, onnx_path: str | Path, sidecar_path: str | Path | None = None,
                 providers: list[str] | None = None):
        import onnxruntime as ort
        self.path = Path(onnx_path)
        self.session = ort.InferenceSession(
            str(self.path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.meta = {}
        if sidecar_path is None:
            cand = self.path.with_suffix(".json")
            sidecar_path = cand if cand.exists() else None
        if sidecar_path:
            self.meta = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
        self.epi = int(self.meta.get("context_len", 22))

    def predict_probs(self, q_padded: np.ndarray, batch: int = 512) -> np.ndarray:
        """``q_padded`` is the uint8 matrix returned by ``features.prepare_recording``."""
        n_epochs = q_padded.shape[1] - self.epi + 1
        if n_epochs <= 0:
            raise ValueError("recording shorter than one context window")
        out = np.empty((n_epochs, 3), dtype=np.float32)
        for lo in range(0, n_epochs, batch):
            hi = min(lo + batch, n_epochs)
            x = np.stack([q_padded[:, j:j + self.epi] for j in range(lo, hi)])
            x = x[:, None, :, :].astype(np.float32)
            logits = self.session.run(None, {self.input_name: x})[0]
            logits = logits - logits.max(axis=1, keepdims=True)
            e = np.exp(logits)
            out[lo:hi] = e / e.sum(axis=1, keepdims=True)
        return out

    def predict(self, img: np.ndarray, cls: np.ndarray | None = None) -> np.ndarray:
        """float (184, T) recording -> per-5 s probabilities."""
        q = features.prepare_recording(img, cls, epi=self.epi)
        return self.predict_probs(q)


def main() -> None:
    import argparse
    import csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--img", required=True, help=".npz cache with img/cls arrays")
    ap.add_argument("--out", default=None, help="optional CSV output")
    args = ap.parse_args()
    img, _ = features.load_cache(args.img)
    model = HypnosONNX(args.onnx)
    probs = model.predict(img)
    labels = probs.argmax(1)
    names = ("REM", "Wake", "NREM")
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "p_rem", "p_wake", "p_nrem", "label"])
            for i, p in enumerate(probs):
                w.writerow([i, f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}",
                            names[int(labels[i])]])
        print(f"wrote {args.out}")
    else:
        print("label counts:", {names[i]: int((labels == i).sum())
                                for i in range(3)})


if __name__ == "__main__":
    main()
