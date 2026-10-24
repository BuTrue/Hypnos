# -*- coding: utf-8 -*-
"""Hypnos v446fusion: the single deployed architecture.

Input  : float tensor (B, 1, 184, context) with 5 s epochs, per-recording
         z-scored and clipped to [-10, 10] (uint8 0..255 in the cache).
Output : logits (B, 3) ordered REM, Wake, NREM.

Two branches share the input:
  1. V446 trunk: three 3x3 conv blocks (8/16/32 channels), frequency-only
     MaxPool, a depthwise temporal conv, squeeze-excitation, and a full-
     resolution flatten head.  This is the original strong baseline.
  2. Band-ratio branch: 16 fixed EEG band powers + EMG level/slope, 12
     pairwise log-ratios, band shape and its first difference; dilated
     depthwise temporal convolutions; multi-window pooling.
The branches are fused in logit space: main + sigmoid(gate) * ratio.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_HEIGHT = 184
N_EPOCH_STEPS = 8
BANDS = [(0.5, 1), (1, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12),
         (12, 14), (14, 16), (16, 18), (18, 20), (20, 25), (25, 30),
         (30, 35), (35, 40), (40, 50)]

_BAND_CACHE: dict[str, torch.Tensor] = {}


class SE1d(nn.Module):
    """Squeeze-excitation over channel descriptors."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        s = x.mean(dim=-1)
        return torch.sigmoid(self.fc2(F.relu(self.fc1(s))))


def _band_buffers():
    if _BAND_CACHE:
        return _BAND_CACHE["idx"], _BAND_CACHE["cnt"]
    f = torch.empty(175)
    f[:100] = (torch.arange(100) + 0.5) * 0.2
    f[100:] = 20.0 + (torch.arange(75) + 0.5) * 0.4
    idx, cnt = [], []
    for lo, hi in BANDS:
        ii = torch.nonzero((f >= lo) & (f < hi)).squeeze(1)
        if len(ii) == 0:
            ii = torch.tensor([int((lo + hi) / 2 / 0.2)])
        idx.append(ii)
        cnt.append(len(ii))
    maxlen = max(len(ii) for ii in idx)
    pad = torch.zeros(len(idx), maxlen, dtype=torch.long)
    for b, ii in enumerate(idx):
        pad[b, :len(ii)] = ii
    _BAND_CACHE["idx"] = pad
    _BAND_CACHE["cnt"] = torch.tensor(cnt, dtype=torch.float32)
    return pad, _BAND_CACHE["cnt"]


def _band_tokens(x: torch.Tensor, idx: torch.Tensor, cnt: torch.Tensor):
    """Mean power in each fixed band plus EMG level and |EMG difference|."""
    eeg = x[:, 0]
    vals = eeg[:, idx, :]
    keep = (torch.arange(idx.shape[1], device=x.device)
            < cnt.to(torch.long)[:, None]).float()
    band = (vals * keep[None, :, :, None]).sum(2) / cnt[None, :, None]
    emg = eeg[:, 176:184, :].mean(1)
    d = torch.zeros_like(emg)
    d[:, 1:] = (emg[:, 1:] - emg[:, :-1]).abs()
    return torch.cat([band, emg[:, None], d[:, None]], dim=1)


class V446Fusion(nn.Module):
    """v446 trunk + parallel band-ratio branch with gated logit fusion."""

    def __init__(self, n_classes: int = 3, epi: int = 22):
        super().__init__()
        self.epi = int(epi)
        self.pool = nn.MaxPool2d((2, 1))
        self.conv1 = nn.Conv2d(1, 8, 3, padding="same")
        self.conv2 = nn.Conv2d(8, 16, 3, padding="same")
        self.conv3 = nn.Conv2d(16, 32, 3, padding="same")
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.tconv = nn.Conv2d(32, 32, (1, 3), padding=(0, 1), groups=32)
        self.tbn = nn.BatchNorm2d(32)
        self.se = SE1d(32, reduction=4)
        self.fc1 = nn.Linear(32 * (IMAGE_HEIGHT // N_EPOCH_STEPS) * self.epi,
                             n_classes)

        idx, cnt = _band_buffers()
        self.register_buffer("band_idx", idx, persistent=False)
        self.register_buffer("band_cnt", cnt, persistent=False)
        self.rconv = nn.Conv1d(60, 64, 7, padding=3)
        self.rbn = nn.BatchNorm1d(64)
        self.rdw2 = nn.Conv1d(64, 64, 5, padding=4, dilation=2, groups=64)
        self.rdw4 = nn.Conv1d(64, 64, 5, padding=8, dilation=4, groups=64)
        self.rpw = nn.Conv1d(64, 64, 1)
        self.rbn2 = nn.BatchNorm1d(64)
        self.rfc = nn.Linear(384, n_classes)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def _ratio_features(self, x: torch.Tensor) -> torch.Tensor:
        t = _band_tokens(x.float(), self.band_idx, self.band_cnt)
        band, emg = t[:, :16], t[:, 16]
        theta = band[:, 3:6].mean(1)
        delta = band[:, 0:3].mean(1)
        alpha = band[:, 5:7].mean(1)
        beta = band[:, 7:10].mean(1)
        gamma = band[:, 11:16].mean(1)
        high = band[:, 10:16].mean(1)
        low = band[:, 0:11].mean(1)
        ratios = torch.stack([theta - delta, theta - alpha, theta - beta,
                              theta - gamma, emg - theta, emg - delta,
                              delta - beta, alpha - beta, gamma - beta,
                              emg - low, theta - high, delta - gamma], dim=1)
        shape = band - band.mean(1, keepdim=True)
        dshape = torch.zeros_like(shape)
        dshape[:, :, 1:] = shape[:, :, 1:] - shape[:, :, :-1]
        return torch.cat([band, ratios, shape, dshape], dim=1)

    def _main(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = F.relu(self.tbn(self.tconv(x)))
        s = self.se(x.mean(dim=3))
        x = x * s[:, :, None, None]
        return self.fc1(torch.flatten(x, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self._main(x)
        z = self._ratio_features(x)
        z = F.relu(self.rbn(self.rconv(z)))
        z = F.relu(self.rdw2(z))
        z = F.relu(self.rdw4(z))
        z = F.relu(self.rbn2(self.rpw(z)))
        c = self.epi // 2
        feats = [z[:, :, c], z.mean(2), z.amax(2), z.std(2),
                 z[:, :, :11].mean(2), z[:, :, -11:].mean(2)]
        ratio_logit = self.rfc(torch.cat(feats, dim=1))
        return main + torch.sigmoid(self.gate) * ratio_logit
