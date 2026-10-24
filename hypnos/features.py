# -*- coding: utf-8 -*-
"""Feature preparation shared by training and inference.

A recording is stored as ``img`` (184, T) float32: 100 log-power bins from
0-20 Hz (0.2 Hz steps), 75 bins from 20-50 Hz (0.4 Hz steps), then 8 EMG
copies in rows 176-183.  ``cls`` is the per-epoch label (0 REM / 1 Wake /
2 NREM, -1 = unlabelled).

Training uses a global per-recording z-score computed over the valid samples
of every channel, then clips to +/-10 and quantises to uint8.  Windows are
edge-padded so that window j covers epochs [j - epi//2, j + epi//2] and the
label at j is the centre (``window_align=label_center``).
"""
from __future__ import annotations

import numpy as np

IMAGE_HEIGHT = 184
N_EPOCH_STEPS = 8
ZSCORE_ABS_MAX = 10.0


def zscore_float(img: np.ndarray, cls: np.ndarray | None = None,
                 abs_max: float = ZSCORE_ABS_MAX) -> np.ndarray:
    """Global per-recording z-score; labels are only used to mask invalid rows."""
    if cls is None:
        valid = np.ones(img.shape[1], dtype=bool)
    else:
        valid = np.asarray(cls) >= 0
    if not valid.any():
        raise ValueError("recording has no valid epochs")
    mean = img[:, valid].mean(axis=1)
    std = img[:, valid].std(axis=1)
    std = np.where(std <= 0, 1.0, std)
    z = (img - mean[:, None]) / std[:, None]
    return np.clip(z, -abs_max, abs_max)


def quantize_uint8(z: np.ndarray, abs_max: float = ZSCORE_ABS_MAX) -> np.ndarray:
    return np.clip((z + abs_max) / (2 * abs_max) * 255.0, 0, 255).astype(np.uint8)


def pad_image(q: np.ndarray, epi: int, causal: bool = False) -> np.ndarray:
    """Edge-pad (184, T) so that window j = padded[:, j:j+epi] is complete."""
    left = epi - 1 if causal else epi // 2
    right = epi - 1 - left
    if left == 0 and right == 0:
        return q
    return np.pad(q, ((0, 0), (left, right)), mode="edge")


def prepare_recording(img: np.ndarray, cls: np.ndarray | None = None,
                      epi: int = 22, causal: bool = False) -> np.ndarray:
    """float image -> padded uint8 feature matrix ready for window extraction."""
    z = zscore_float(img, cls)
    return pad_image(quantize_uint8(z), epi, causal=causal)


def load_cache(path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as d:
        return d["img"].astype(np.float32), d["cls"].astype(np.int64)
