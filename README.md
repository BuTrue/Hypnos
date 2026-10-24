# Hypnos — mouse sleep staging

Hypnos is a lightweight single-model classifier for mouse EEG/EMG sleep
staging (REM / Wake / NREM). This repository contains the architecture, the
training and decoding method, a documented 5-fold benchmark against nine
published sleep-staging baselines, and the deployed ONNX model.

## Headline results

Protocol: 31 private HYX recordings / 16 mouse groups, mouse-disjoint 5-fold
cross-validation, 5 s epochs, raw argmax (no HMM or prior bias). All models
were trained on the same folds and labels (the LGBM-G2 variant uses a
human-pretrained YASA feature trunk). External neural models are faithful
PyTorch ports, not the authors' original code.

| model | family | REM F1 | Wake F1 | NREM F1 | mean F1 | sum F1 |
|---|---|---|---|---|---|---|
| **Hypnos-V446 (legacy)** | project CNN, 45 s context | **0.7108** | 0.9359 | 0.9284 | **0.8584** | **2.5752** |
| **Hypnos-V448 (legacy)** | project CNN, 45 s context | 0.7047 | 0.9344 | 0.9277 | 0.8556 | 2.5668 |
| SleepGPT | time–frequency Transformer | 0.6994 | 0.9224 | 0.9110 | 0.8443 | 2.5328 |
| SleepTransformer | two-level Transformer | 0.6940 | 0.9193 | 0.9116 | 0.8416 | 2.5249 |
| **Hypnos-v446fusion (CV5, 10EP)** | project CNN + band-ratio, 110 s | 0.6535 | 0.9203 | 0.9106 | 0.8281 | 2.4843 |
| **Hypnos-v446fusion (CV5-E9, 10EP)** | project CNN + band-ratio, 45 s | 0.6246 | 0.9285 | 0.9174 | 0.8235 | 2.4705 |
| **Hypnos-v446fusion (CV5-30EP, pilot)** | project CNN + band-ratio, 110 s | 0.6276 | 0.9093 | 0.8836 | 0.8068 | 2.4205 |
| DeepSleepNet | CNN + BiLSTM | 0.5993 | 0.8852 | 0.8420 | 0.7755 | 2.3264 |

> The **CV5-30EP** row follows the deployed 30-epoch cosine schedule but stops
> within the first 10 epochs (the deployment checkpoint is epoch 7); it is a
> pilot, not a full 30-epoch 5-fold run. It does not beat the 10-epoch model.
> With HMM postprocessing the modern Hypnos rows gain +0.03 to +0.05 REM, but
> the ranking does not change: see
> [`benchmark/results/hmm_comparison.md`](benchmark/results/hmm_comparison.md).

Full 14-model table (AccuSleep, AttnSleep, LGBM, U-Sleep, TinySleepNet, …):
[`benchmark/results/benchmark_master.md`](benchmark/results/benchmark_master.md).
Resource comparison (parameters, size, latency, memory, training time, 24 h
inference): [`benchmark/results/resources.md`](benchmark/results/resources.md).

![benchmark](benchmark/figures/benchmark_summary.png)
![F1](benchmark/figures/f1_by_class.png)

## Deployed model

- `models/LAB-CV5-CORE-30EP.onnx` — v446fusion, **88,607 parameters**, 5 s
  epochs, 110 s context, 0.389 MB.
- ONNX CPU latency **1.12 ms per 5 s epoch**; GPU peak memory **10 MB**;
  a 24 h file takes **19.3 s** of model compute (25.1 s end-to-end including
  feature extraction in the TrueSleep app).
- Sidecar: `models/LAB-CV5-CORE-30EP.json` (class order REM/Wake/NREM,
  `window_align=label_center`, `zscore_abs_max=10.0`).
- Model card: [`models/model_card.md`](models/model_card.md).
- Honest generalisation estimates: fold-0 held-out REM F1 **0.7342** (5 s core)
  / **0.7395** (10 s core); full 5-fold OOF REM F1 **0.7231** (10 s core).
  See [`docs/PRODUCTION.md`](docs/PRODUCTION.md). The model is trained on all
  31 recordings, so in-sample numbers are not used for comparison.
- **30-epoch 5-fold pilot:** raw REM 0.6276 / HMM+prior 0.6568, below the
  10-epoch model (raw 0.6535 / HMM 0.6822). The deployed checkpoint (epoch 7)
  is good on its held-out fold, but the 30-epoch recipe is not better in full
  cross-validation.

## Method in one page

- **Features**: 100 log-power bins 0–20 Hz, 75 bins 20–50 Hz, 8 EMG rows →
  `184 × 22` input (110 s context), per-recording global z-score, no
  augmentation.
- **Architecture**: V446 conv trunk + full-resolution flatten head, plus a
  parallel 16-band ratio branch; gated logit fusion.
- **Training**: class-weighted cross-entropy, SGD + cosine, batch 64; core
  training excludes ±1 epoch around label changes.
- **Decoding**: fold to 10 s, 4.5 % REM prior, Viterbi λ=0.5 with a transition
  matrix fitted on training recordings.
- Test mice are never in training. No test-label statistics are used.

Full details and runnable code: [`docs/METHODS.md`](docs/METHODS.md),
[`hypnos/`](hypnos).

```bash
pip install -r requirements.txt
python -m hypnos.infer --onnx models/LAB-CV5-CORE-30EP.onnx --img cache/16.npz
```

## Benchmark

- Protocol and full table: [`docs/BENCHMARK.md`](docs/BENCHMARK.md)
- Machine-readable results: [`benchmark/results/`](benchmark/results)
- Statistical significance (paired bootstrap): [`benchmark/results/significance.md`](benchmark/results/significance.md)
- External model sources, licenses and port notes:
  [`docs/EXTERNAL_MODELS.md`](docs/EXTERNAL_MODELS.md)
- Reproduction/figure code: [`benchmark/code/`](benchmark/code)

## Repository layout

```
hypnos/            v446fusion, features, decoding, training, ONNX inference
models/            deployed ONNX model, sidecar, model card
docs/              methods, benchmark, production, external models
benchmark/         results, figures, chart/aggregation code
```

## Data

The HYX EDF/DB3 recordings are **not** included. Native DB3 annotations are
10 s rows repeated to 5 s; the main benchmark scores 5 s epochs, while the
production numbers also report 10 s scoring. Without data access the exact
numbers cannot be reproduced, but the model, feature format and evaluation
code are public.

## Citation

If you use this benchmark, please cite the Hypnos project and the original
papers of the external models listed in
[`docs/EXTERNAL_MODELS.md`](docs/EXTERNAL_MODELS.md). External code and weights
are not redistributed; their licenses remain in force.

## License

Code: MIT (see [`LICENSE`](LICENSE)). The released ONNX weights are provided
for research use.
