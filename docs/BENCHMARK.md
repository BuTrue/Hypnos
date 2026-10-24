# Benchmark

## Protocol

- **Data**: 31 HYX recordings, 16 mouse groups; 3 classes (REM / Wake / NREM).
- **Split**: the same mouse-disjoint 5 folds for every model (table in
  [`METHODS.md`](METHODS.md)).
- **Epoch**: 5 s scoring. Native DB3 annotations are 10 s rows; the 5 s label is
  the repeated 10 s label. No epoch from a test mouse is ever trained on.
- **Training**: every model is trained on the training folds with the same
  labels; the LGBM-G2 variant uses a human-pretrained YASA feature trunk.
  External neural models are faithful PyTorch ports written for this benchmark;
  see [`EXTERNAL_MODELS.md`](EXTERNAL_MODELS.md).
- **Scoring**: raw argmax at 5 s, **no HMM, no prior bias, no transition
  smoothing**. F1 is computed per recording, averaged over recordings, then over
  the five folds. `std R` is the population standard deviation of the five
  per-fold REM F1 means.
- **Selection**: one validation-based checkpoint per fold; test groups are
  scored once.

## Main table

| # | model | family | context | REM F1 | Wake F1 | NREM F1 | mean F1 | sum F1 | std REM | params |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Hypnos-V446 (legacy)** | project CNN | 9 (45 s) | **0.7108** | 0.9359 | 0.9284 | **0.8584** | **2.5752** | 0.0900 | — |
| 2 | **Hypnos-V448 (legacy)** | project CNN | 9 (45 s) | 0.7047 | 0.9344 | 0.9277 | 0.8556 | 2.5668 | 0.0920 | — |
| 3 | SleepGPT | time–frequency Transformer | 9×20 | 0.6994 | 0.9224 | 0.9110 | 0.8443 | 2.5328 | 0.0868 | 1,670,371 |
| 4 | SleepTransformer | two-level Transformer | 9×20 | 0.6940 | 0.9193 | 0.9116 | 0.8416 | 2.5249 | 0.0387 | 3,353,139 |
| 5 | **Hypnos-v446fusion (CV5, 10EP)** | project CNN + band-ratio | 22 (110 s) | 0.6535 | 0.9203 | 0.9106 | 0.8281 | 2.4843 | 0.0536 | 88,607 |
| 6 | **Hypnos-v446fusion (CV5-E9, 10EP)** | project CNN + band-ratio | 9 (45 s) | 0.6246 | 0.9285 | 0.9174 | 0.8235 | 2.4705 | 0.1421 | 59,903 |
| 7 | **Hypnos-v446fusion (CV5-30EP, pilot)** | project CNN + band-ratio | 22 (110 s) | 0.6276 | 0.9093 | 0.8836 | 0.8068 | 2.4205 | 0.0938 | 88,607 |
| 8 | DeepSleepNet | CNN + BiLSTM | 20×5 s | 0.5993 | 0.8852 | 0.8420 | 0.7755 | 2.3264 | 0.0781 | 7,628,547 |
| 9 | LGBM-G2 (YASA transfer) | LightGBM | 5 s | 0.4982 | 0.8926 | 0.8706 | 0.7538 | 2.2615 | 0.0797 | — |
| 10 | AccuSleep | compact CNN | 9 | 0.3540 | 0.9149 | 0.9082 | 0.7257 | 2.1771 | 0.1734 | 14,835 |
| 11 | U-Sleep (lightweight) | U-Net | 20×5 s | 0.2974 | 0.7921 | 0.8326 | 0.6407 | 1.9221 | 0.0320 | 38,387 |
| 12 | LGBM-153feat | LightGBM | 5 s | 0.2548 | 0.8575 | 0.8039 | 0.6387 | 1.9162 | 0.0688 | — |
| 13 | AttnSleep | SE-CNN + TCE | 9 | 0.2529 | 0.7552 | 0.8310 | 0.6130 | 1.8391 | 0.0459 | 518,003 |
| 14 | TinySleepNet | CNN + GRU | 20×5 s | 0.0534 | 0.6224 | 0.4298 | 0.3685 | 1.1056 | 0.0688 | 421,123 |

> **CV5-30EP is a pilot, not a full 30-epoch run.** It follows the deployed
> 30-epoch cosine schedule but stops within the first 10 epochs (patience 3;
> the deployment checkpoint is epoch 7). Under current GPU load a full
> 30-epoch × 5-fold run would take >10 h. The pilot does **not** beat the
> 10-epoch CV5 row.
>
> CV5-E9 uses the same 45 s context as the legacy rows and a 10-epoch budget,
> isolating the effect of context length and architecture.

![sum F1 and trade-offs](../benchmark/figures/benchmark_summary.png)

![per-class F1](../benchmark/figures/f1_by_class.png)

## Reading the table

- **Our strongest row is not a modern fusion row.** It is the legacy V446/V448
  pair (`EPI=9`, 45 s context), trained under the same 5-fold protocol as the
  external models. The modern `v446fusion` rows (110 s or 45 s context) are
  included for completeness; none of them beats V446 on raw 5 s F1.
- The deployed model (`LAB-CV5-CORE-30EP`) is trained on all 31 recordings.
  Its 5-fold generalisation estimate is the CV5-30EP pilot row above; the
  in-sample numbers live in [`PRODUCTION.md`](PRODUCTION.md), separate from
  this table.
- **REM is the discriminating class.** Wake and NREM F1 are saturated near 0.9
  for almost every model; ranking by mean F1 is mostly ranking by REM F1.
  `sum F1` is shown because it is the metric used internally.
- **Context length matters.** Models with 45–110 s context (Hypnos, SleepGPT,
  SleepTransformer) outperform single-epoch or short-context models. The
  lightweight U-Sleep port is an exception: it is fast but weak on REM.
- **Variance is large.** Legacy Hypnos cross-fold REM std is 0.09; the CV5-E9
  row is 0.14. A 0.01 difference between adjacent rows is not meaningful with
  31 recordings.

## HMM postprocessing

The main table uses raw argmax. Folding to 10 s and applying a transition
matrix (plus a 4.5 % REM prior for Hypnos) changes the ranking only slightly:

| model | raw 5 s REM | HMM REM | delta | setting |
|---|---|---|---|---|
| Hypnos-V446 (legacy) | 0.7108 | 0.7101 | -0.0007 | lambda 0.3 |
| Hypnos-V448 (legacy) | 0.7047 | 0.7071 | +0.0024 | lambda 0.3 |
| SleepGPT | 0.6994 | **0.7201** | +0.0207 | lambda 1.0 |
| SleepTransformer | 0.6940 | 0.7046 | +0.0106 | lambda 1.0 |
| Hypnos-v446fusion (CV5, 10EP) | 0.6535 | 0.6822 | +0.0287 | lambda 0.5 + 4.5 % prior |
| Hypnos-v446fusion (CV5-E9) | 0.6246 | 0.6782 | +0.0536 | lambda 0.5 + 4.5 % prior |
| Hypnos-v446fusion (CV5-30EP, pilot) | 0.6276 | 0.6568 | +0.0292 | lambda 0.5 + 4.5 % prior |
| DeepSleepNet | 0.5993 | 0.6292 | +0.0299 | lambda 1.0 |
| AccuSleep | 0.3540 | 0.3374 | -0.0166 | 10 s |
| LGBM-153feat | 0.2548 | 0.3744 | +0.1196 | 10 s |

- HMM helps 8 of 10 models; the gain is largest for weak models and nearly
  zero for the already-strong legacy V446.
- On the modern Hypnos models the 4.5 % REM prior supplies most of the gain;
  HMM alone (lambda 0.5) adds about +0.01 REM.
- Lambda 1.0 over-smooths the 22-epoch models (10EP 0.6609 vs 0.6697 at
  lambda 0.5; 30EP 0.6363 vs 0.6415).
- Even with HMM, the best modern Hypnos row (0.6822) stays below SleepGPT+HMM
  (0.7201) and SleepTransformer+HMM (0.7046). The best Hypnos HMM row is still
  legacy V446 (0.7101).
- Full variant tables: [`results/hmm_comparison.md`](../benchmark/results/hmm_comparison.md),
  `cv5_10ep_eval.md`, `cv5_30ep_eval.md`, `epi9_eval.md`.

## Resource comparison

Parameters, weight size, static parameter memory, measured latency, GPU peak
memory, training time and 24 h inference time are in
[`results/resources.md`](../benchmark/results/resources.md).

Key point: the deployed Hypnos model is **88,607 parameters (0.39 MB ONNX)**,
runs a 5 s epoch in **1.12 ms on CPU** (ONNX Runtime, batch 1), and needs
**10 MB of GPU memory** at batch 1. It is roughly 38× smaller than
SleepTransformer and 86× smaller than DeepSleepNet.

## Limitations

- The HYX corpus is private; the numbers cannot be reproduced without data
  access. Code and feature format are public.
- Native labels are 10 s. 5 s metrics inherit the 10 s annotation granularity;
  "5 s performance" is not independent 5 s human scoring.
- Class imbalance is severe (REM ≈ 5–10 % of epochs). REM F1 is therefore noisy
  and sensitive to a handful of recordings.
- External models are re-implementations, not the authors' original code; port
  fidelity varies. They were trained with the same budget policy, not tuned
  per model.
- Single seed per model/fold. No confidence intervals are reported for external
  models.
- YASA (human pretrained) was evaluated on two recordings only and is excluded
  from the table; it predicts almost everything as NREM (domain gap).
