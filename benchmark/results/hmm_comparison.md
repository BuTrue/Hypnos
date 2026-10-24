# HMM postprocessing comparison

Raw column is 5 s argmax (the main-table protocol). HMM columns fold two 5 s
predictions into one 10 s block and decode with a transition matrix fitted on
the training recordings of each fold.

| model | raw 5 s REM | HMM REM | delta | HMM setting | source |
|---|---|---|---|---|---|
| Hypnos-V446 (legacy) | **0.7108** | 0.7101 | -0.0007 | 10 s, lambda 0.3 | 2026-09 benchmark |
| Hypnos-V448 (legacy) | 0.7047 | 0.7071 | +0.0024 | 10 s, lambda 0.3 | 2026-09 benchmark |
| SleepGPT | 0.6994 | **0.7201** | +0.0207 | 10 s, lambda 1.0 | 2026-09 benchmark |
| SleepTransformer | 0.6940 | 0.7046 | +0.0106 | 10 s, lambda 1.0 | 2026-09 benchmark |
| Hypnos-v446fusion (CV5, 10EP) | 0.6535 | 0.6822 | +0.0287 | 10 s, lambda 0.5 + 4.5% prior | this repo |
| Hypnos-v446fusion (CV5-E9, 10EP) | 0.6246 | 0.6782 | +0.0536 | 10 s, lambda 0.5 + 4.5% prior | this repo |
| Hypnos-v446fusion (CV5-30EP, pilot) | 0.6276 | 0.6568 | +0.0292 | 10 s, lambda 0.5 + 4.5% prior | this repo |
| DeepSleepNet | 0.5993 | 0.6292 | +0.0299 | 10 s, lambda 1.0 | 2026-09 benchmark |
| AccuSleep | 0.3540 | 0.3374 | -0.0166 | 10 s | 2026-09 benchmark |
| LGBM-153feat | 0.2548 | 0.3744 | +0.1196 | 10 s | 2026-09 benchmark |

## No-prior HMM on the Hypnos variants

| model | raw 5 s | HMM lambda 0.5 | HMM lambda 1.0 | HMM lambda 0.5 + 4.5% prior |
|---|---|---|---|---|
| CV5, 10EP | 0.6535 | 0.6697 | 0.6609 | **0.6822** |
| CV5-E9, 10EP | 0.6246 | 0.6716 | 0.6749 | 0.6782 |
| CV5-30EP, pilot | 0.6276 | 0.6415 | 0.6363 | **0.6568** |

Core REM (10 s, boundary epochs excluded) for the deployed decoder
(lambda 0.5 + prior): 10EP 0.6932 -> 0.7222; E9 0.6460 -> 0.7005;
30EP pilot 0.6607 -> 0.6865.

## What this shows

- HMM helps 8 of 10 models. The gain is largest where raw predictions are
  weakest (LGBM +0.12, CV5-E9 +0.054) and nearly zero for the already-strong
  legacy V446 (-0.001).
- On the modern Hypnos models, the 4.5% REM prior supplies most of the gain;
  HMM alone (lambda 0.5) adds about +0.01 REM.
- Lambda 1.0 over-smooths the 22-epoch models (10EP 0.6609 vs 0.6697 at
  lambda 0.5; 30EP 0.6363 vs 0.6415).
- Even with HMM, the best modern Hypnos row (CV5 10EP, 0.6822) stays below
  SleepGPT+HMM (0.7201) and SleepTransformer+HMM (0.7046). The best Hypnos
  HMM row remains legacy V446 (0.7101).
- The CV5-30EP row is the 30-epoch cosine recipe stopped within the first
  10 epochs (best deployment checkpoint is epoch 7); it is a pilot, not a
  full 30-epoch 5-fold run. It does not beat the 10EP model.

Detailed per-variant tables: `cv5_10ep_eval.md`, `cv5_30ep_eval.md`,
`epi9_eval.md`.
