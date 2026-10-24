# Methods

## 1. Task and data

Hypnos classifies 5 s epochs of mouse EEG/EMG into **REM**, **Wake** and
**NREM**. All experiments use the private HYX corpus:

- 31 paired EEG/EMG recordings from 16 mouse groups;
- native DB3 annotations are 10 s rows, repeated to 5 s for model training and
  5 s scoring (the 10 s protocol is used for the primary deployment numbers);
- recordings are split by **mouse group**, never by epoch. No epoch from a test
  mouse appears in training or validation.

The five folds are identical to the ones used by every model in the benchmark:

| fold | held-out test groups | validation groups |
|---|---|---|
| 0 | NO5 (4 recs), M16, M22 | NO6, NO7 |
| 1 | NO9 (4 recs), ARC17, DMH | NO17, NO33 |
| 2 | NO6, NO17, M3 (2 recs) | NO7, NO8 |
| 3 | NO7, NO8, PVN8 (2 recs) | ARC7 (2 recs), PVN9 (2 recs) |
| 4 | NO33 (2 recs), ARC7 (2 recs), PVN9 (2 recs), M4 | M16, M22 |

Test recordings are scored once, after model selection on the validation
groups.

## 2. Input representation

Each 5 s epoch is converted to a 184-row feature vector:

| rows | content | resolution |
|---|---|---|
| 0–99 | log power 0–20 Hz | 0.2 Hz |
| 100–174 | log power 20–50 Hz | 0.4 Hz |
| 176–183 | 8 EMG copies | — |

Training normalisation is a **global per-recording z-score**: for every feature
row, subtract the mean and divide by the standard deviation over the epochs of
that recording (labels are used only to mask invalid rows). The result is
clipped to ±10 and quantised to uint8. No test-label statistics are used.

A model input is a context window of `EPI` epochs. The window is edge-padded by
`EPI//2` on the left and `EPI-1-EPI//2` on the right, so window `j` is centred on
epoch `j` (`window_align=label_center`). The deployed model uses `EPI=22`
(110 s); the legacy benchmark models used `EPI=9` (45 s).

## 3. Architecture: v446fusion

`v446fusion` (88,607 parameters at `EPI=22`) is a single network with two
branches whose logits are added:

```
logits = v446_trunk(x) + sigmoid(gate) * band_ratio_branch(x)
```

### 3.1 V446 trunk

- three 3×3 convolutions: 1→8→16→32 channels, each followed by BatchNorm and
  ReLU;
- frequency-only `MaxPool(2,1)` after each block (184→23 rows, time preserved);
- depthwise temporal convolution (kernel 3) + BatchNorm;
- squeeze-excitation over the 32 channel descriptors;
- flatten and a linear head `32 × 23 × EPI → 3`.

The full-resolution flatten head is the original V446 design; it keeps the
frequency structure that band-pooling destroys.

### 3.2 Band-ratio branch

- 16 fixed EEG band powers (0.5–50 Hz) + EMG level + |EMG difference|;
- 12 pairwise log-ratios (theta/delta, theta/alpha, …, EMG/low);
- band shape and its first temporal difference → 60 input channels;
- Conv1d(60→64, k=7), two depthwise dilated convolutions (k=5, dilation 2 and
  4), pointwise 64→64;
- pooling at the centre, mean, max, std, first 11 and last 11 epochs → 384
  features → Linear(384→3).

The gate starts at 0 (ratio branch off) and is learned end-to-end.

## 4. Training

| item | value |
|---|---|
| loss | class-weighted cross-entropy, weight `N / (3 × count_c)` |
| optimiser | SGD, lr 0.005, momentum 0.9, weight decay 1e-4 |
| schedule | cosine over all steps |
| batch | 64 |
| gradient clipping | 1.0 |
| augmentation | none in the delivered model |
| CV epochs | 10 per fold (5-fold protocol) |
| deployment schedule | 30 epochs; best checkpoint at epoch 7 |

**Core training** (`drop_boundary=1`) excludes epochs within ±1 of a label
change from the training index. Boundaries are ambiguous and their labels are
the noisiest part of DB3; removing them improved the transition-adjacent
"core" metrics. Validation and test scoring always use all epochs, including
boundaries.

Model selection uses validation 10 s core REM F1 after decoding. The deployment
model is a single seed (seed 42); weight averaging of two seeds collapsed REM
probability and was rejected.

## 5. Decoding

The deployed pipeline:

1. fold two consecutive 5 s probability vectors into one 10 s block (mean);
2. add a global REM log-probability bias chosen so the predicted REM rate equals
   4.5 % (the corpus prior);
3. Viterbi decoding with `lambda=0.5` and a transition matrix fitted on the
   training recordings.

The benchmark's G1 protocol (used for the external models and for the fair
Hypnos row) reports **raw argmax at 5 s without steps 1–3**, so the comparison
is not inflated by postprocessing.

## 6. Metrics

- **F1** is computed per recording and then averaged over recordings; folds are
  averaged for the final number.
- **mean F1** is the unweighted mean of REM/Wake/NREM F1; **sum F1** is their sum.
- **core** metrics exclude epochs within ±1 of a label transition (same rule as
  core training). This isolates steady-state staging from boundary ambiguity.

## 7. Reproduction

The public code is a minimal, dependency-light version of the pipeline.

```bash
pip install -r requirements.txt

# train 5 folds on cached features
python -m hypnos.train --data cache --folds folds.json --out runs \
    --epochs 10 --epi 22 --drop-boundary 0

# run the deployed ONNX model on one cached recording
python - <<'PY'
from hypnos import features, infer, decode

img, y = features.load_cache("cache/16.npz")       # y is only used for scoring
model = infer.HypnosONNX("models/LAB-CV5-CORE-30EP.onnx")
probs = model.predict(img)                          # no labels needed at inference

_, train_y = features.load_cache("cache/ARC17ds.npz")
trans = decode.fit_transition(train_y)              # transition prior from training data

p10, y10 = decode.fold_to_10s(probs, y, 5.0)
pred = decode.decode(p10, trans)
print(decode.metrics(y10, pred))
PY
```

The dataset itself cannot be redistributed; the cache format is documented in
`hypnos/features.py`.
