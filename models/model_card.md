# Model card — LAB-CV5-CORE-30EP

## Identity

- architecture: `v446fusion` (single model, no ensemble, no distillation)
- 5.0 s epochs, context 22 epochs = 110 s, input `(B, 1, 184, 22)`
- parameters: 88,607
- weights: `LAB-CV5-CORE-30EP.onnx` (0.389 MB), sidecar `LAB-CV5-CORE-30EP.json`
- single seed (42); 30-epoch cosine schedule, best checkpoint at epoch 7

## Training data

- all 31 paired recordings of the private HYX corpus (16 mouse groups)
- 470,865 core-training windows
- global per-recording z-score, no augmentation
- class-weighted cross-entropy; SGD lr 0.005, momentum 0.9, wd 1e-4, batch 64
- core training excludes epochs within ±1 of a label change

## Generalisation estimates

These are leakage-free: fold 0 held out three mouse groups; the 5-fold OOF
model (10 epochs) never saw its test groups.

| estimate | grid | REM | Wake | NREM | mean | sum |
|---|---|---|---|---|---|---|
| fold-0 held-out, 30 ep | 5 s core | 0.7342 | 0.9631 | 0.9524 | 0.8832 | 2.6497 |
| fold-0 held-out, 30 ep | 10 s core | 0.7395 | 0.9765 | 0.9668 | 0.8943 | 2.6828 |
| 5-fold OOF, 10 ep | 10 s full | 0.6828 | 0.9232 | 0.9126 | 0.8395 | 2.5186 |
| 5-fold OOF, 10 ep | 10 s core | 0.7231 | 0.9744 | 0.9748 | 0.8908 | 2.6723 |
| 30-epoch pilot, raw 5 s | 5 s full | 0.6276 | 0.9093 | 0.8836 | 0.8068 | 2.4205 |
| 30-epoch pilot, HMM+prior | 10 s full | 0.6568 | 0.9094 | 0.8836 | 0.8166 | 2.4498 |

The 30-epoch pilot follows the deployed cosine schedule but stops within the
first 10 epochs (deployment checkpoint epoch 7). It does not beat the 10-epoch
model; fold 2 is the failure case (raw REM 0.4817). Treat it as a negative
result, not as an improvement.

For reference, fold-0 full-label metrics are 0.6968 / 0.9204 / 0.9010 at 5 s
(mean 0.8394, sum 2.5181) and 0.7118 / 0.9221 / 0.9017 at 10 s (mean 0.8452,
sum 2.5356).

## In-sample checks (delivery model sees all 31 recordings)

| grid | REM | mean | sum |
|---|---|---|---|
| 5 s full | 0.8862 | 0.9307 | 2.7920 |
| 5 s core | 0.9472 | 0.9751 | 2.9253 |
| 10 s full | 0.9002 | 0.9359 | 2.8075 |
| 10 s core | 0.9659 | 0.9852 | 2.9557 |

These are **not** generalisation estimates; use the fold-0 / OOF numbers above.

## Deployment

- ONNX input: float32 `(B, 1, 184, 22)`, feature scale 0–255.
- Output: logits, class order REM / Wake / NREM.
- ONNX CPU latency: 1.12 ms per 5 s epoch (batch 1).
- GPU peak memory: 10 MB (batch 1).
- 24 h file: ≈ 19.3 s of model compute; 25.1 s end-to-end on a 24.4 h
  recording including feature extraction and postprocessing.
- Decoding: fold to 10 s, 4.5 % REM prior, Viterbi λ=0.5 with a transition
  matrix fitted on training recordings.

## Limitations

- Private corpus of 16 mouse groups; cross-lab generalisation untested.
- Native labels are 10 s DB3 rows repeated to 5 s; adjacent 5 s predictions are
  not independent ground truth.
- Single seed. A two-seed weight average collapsed REM probability and was
  rejected.
- REM is a minority class; per-recording REM F1 is noisy.

## Intended use

Research use for mouse sleep staging. Not a medical device and not validated for
human sleep or clinical decisions.
