# Deployed model

## Identity

| item | value |
|---|---|
| name | `LAB-CV5-CORE-30EP` |
| architecture | `v446fusion` (single model, no ensemble, no distillation) |
| parameters | 88,607 |
| epoch | 5.0 s |
| context | 22 epochs = 110 s |
| input | `(B, 1, 184, 22)` float32, 0–255 scale |
| output | logits, class order REM / Wake / NREM |
| weights | `models/LAB-CV5-CORE-30EP.onnx` (0.389 MB) + `.json` sidecar |
| training | all 31 HYX recordings, 470,865 core windows |
| schedule | 30-epoch cosine; selected checkpoint epoch 7 |
| seed | 42, single seed |

Training details, normalisation and decoding are in
[`METHODS.md`](METHODS.md). The model card with the full evaluation is
[`models/model_card.md`](../models/model_card.md).

## How well does it generalise?

The deployment model is trained on **all 31 recordings**, so any score on those
recordings is in-sample. Two leakage-free estimates are available:

### Fold-0 held-out estimate (30-epoch protocol)

Fold 0 test groups (NO5, M16, M22) were never trained on; checkpoint selection
used NO6/NO7.

| grid | REM | Wake | NREM | mean | sum |
|---|---|---|---|---|---|
| 5 s full | 0.6968 | 0.9204 | 0.9010 | 0.8394 | 2.5181 |
| 5 s core | **0.7342** | 0.9631 | 0.9524 | **0.8832** | 2.6497 |
| 10 s full | 0.7118 | 0.9221 | 0.9017 | 0.8452 | 2.5356 |
| 10 s core | **0.7395** | 0.9765 | 0.9668 | **0.8943** | 2.6828 |

### Full 5-fold OOF estimate (10-epoch CV5 model)

The deployed 30-epoch model is fine-tuned from the same recipe; the fully
cross-validated estimate comes from the 10-epoch 5-fold run (10 s scoring,
postprocessed):

| grid | REM | Wake | NREM | mean | sum |
|---|---|---|---|---|---|
| 10 s full | 0.6828 | 0.9232 | 0.9126 | 0.8395 | 2.5186 |
| 10 s core | 0.7231 | 0.9744 | 0.9748 | 0.8908 | 2.6723 |

Use the fold-0 number as the realistic single-fold estimate and the OOF number
as the cross-mouse estimate.

## In-sample checks (for reference only)

The delivery model sees all 31 recordings, so these numbers are **not**
generalisation estimates.

| grid | REM | mean | sum |
|---|---|---|---|
| 5 s full | 0.8862 | 0.9307 | 2.7920 |
| 5 s core | 0.9472 | 0.9751 | 2.9253 |
| 10 s full | 0.9002 | 0.9359 | 2.8075 |
| 10 s core | 0.9659 | 0.9852 | 2.9557 |

TrueSleep end-to-end, five recordings, iters=0:

| recording | 5 s full REM | 5 s full mean | 5 s core REM | 5 s core mean | 10 s core REM |
|---|---|---|---|---|---|
| NO33 | 0.8095 | 0.9086 | 0.8886 | 0.9572 | 0.8763 |
| ARC | 0.9117 | 0.9330 | 0.9562 | 0.9678 | 0.9519 |
| M3-46 | 0.8944 | 0.9335 | 0.9475 | 0.9744 | 0.9733 |
| DMH | 0.8887 | 0.9253 | 0.9581 | 0.9798 | 0.9569 |
| 22 | 0.8838 | 0.9341 | 0.9236 | 0.9700 | 0.8984 |

## Speed and memory

Measured on dual Xeon E5-2696 v3 + RTX 3060 12 GB:

| item | value |
|---|---|
| ONNX CPU, batch 1 | **1.12 ms / 5 s epoch** |
| PyTorch CPU, 8 threads | 4.33 ms / epoch |
| PyTorch GPU, batch 1 | 5.01 ms / epoch |
| GPU peak, batch 1 | 10.0 MB |
| PyTorch weights | 0.373 MB |
| ONNX weights | 0.389 MB |
| 24 h file, model only | 17,280 epochs × 1.12 ms ≈ **19.3 s** |
| 24.4 h DMH, TrueSleep end-to-end | **25.1 s** (features + inference + postprocessing) |

CPU latency is lower than GPU because the batch-1 network is tiny and kernel
launch dominates on GPU.

## ONNX sidecar

`models/LAB-CV5-CORE-30EP.json`:

| field | value | meaning |
|---|---|---|
| `epoch_sec` | 5.0 | feature epoch |
| `context_len` | 22 | 110 s context |
| `class_order` | REM, Wake, NREM | output order |
| `window_align` | `label_center` | window j is centred on epoch j |
| `zscore_abs_max` | 10.0 | quantisation clip |
| `mixture` | 184 mu / sigma | global initialisation for the app's self-calibration |
| `mixture_init` | `global` | start self-calibration from dataset stats |

The app runs a mixture z-score with per-recording self-calibration. The public
`hypnos.infer` wrapper uses the training-time global per-recording z-score. The
PyTorch definition in `hypnos/models.py` and the released ONNX file were
verified to agree: maximum probability difference 2.9e-6 and identical argmax
on 128 epochs of a held-out recording (same preprocessing).

## Usage

```bash
python -m hypnos.infer --onnx models/LAB-CV5-CORE-30EP.onnx \
    --img cache/16.npz --out predictions.csv
```

or from Python:

```python
from hypnos import features, infer

img, _ = features.load_cache("cache/16.npz")
model = infer.HypnosONNX("models/LAB-CV5-CORE-30EP.onnx")
probs = model.predict(img)          # (T, 3) for REM / Wake / NREM
```

See [`METHODS.md`](METHODS.md) for decoding and scoring.

## Caveats

- The 30-epoch schedule selected its best checkpoint at epoch 7; no later epoch
  improved validation (patience 10 would have stopped at epoch 17).
- Single seed. A two-seed weight average collapsed the REM probability and was
  rejected.
- Trained on a private corpus of 16 mouse groups; cross-lab generalisation is
  untested.
- The model outputs 5 s labels, but the native DB3 annotations are 10 s. Do not
  interpret adjacent 5 s predictions as independent ground truth.
