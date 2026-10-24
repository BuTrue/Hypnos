# Resource comparison

Hardware: dual Xeon E5-2696 v3 (36 C / 72 T), 48 GB RAM, RTX 3060 12 GB.
Latency is batch=1 on one 5 s context window; Hypnos uses ONNX Runtime CPU (1.12 ms) and PyTorch GPU (5.01 ms). Parameter memory is 4 bytes x parameters (fp32 weights only; activations are not included). External-model latency was not re-measured on this machine; the 24 h column comes from the original benchmark run on a ~24 h recording.

| model | params | weights (MB) | param memory (MB) | latency 5 s (ms) | GPU peak (MB) | train (min/fold) | 24 h inference (s) |
|---|---|---|---|---|---|---|---|
| Hypnos-V446 (legacy) | — | 0.24 | — | — | — | 40.0 | 6.3 |
| Hypnos-V448 (legacy) | — | 0.24 | — | — | — | 40.0 | 6.3 |
| SleepGPT | 1670371 | 6.72 | 6.681 | — | — | 38.0 | 73.4 |
| SleepTransformer | 3353139 | 13.43 | 13.413 | — | — | 32.54 | 70.4 |
| Hypnos-v446fusion (CV5, OOF) | 88607 | 0.373 | 0.354 | 1.12 | 10.0 | 18.12 | 19.3 |
| DeepSleepNet | 7628547 | 30.55 | 30.514 | — | — | 90.08 | 161.0 |
| LGBM-G2 (YASA transfer) | — | 14.51 | — | — | — | 0.15 | — |
| AccuSleep | 14835 | 0.07 | 0.059 | — | — | 8.87 | 6.3 |
| U-Sleep (lightweight) | 38387 | 0.18 | 0.154 | — | — | 49.37 | 1.8 |
| LGBM-153feat | — | 14.51 | — | — | — | 12.54 | 44.7 |
| AttnSleep | 518003 | 2.11 | 2.072 | — | — | 80.0 | — |
| TinySleepNet | 421123 | 1.7 | 1.684 | — | — | 10.39 | — |

Notes

- Hypnos CV5: 88,607 parameters; 0.373 MB PyTorch weights, 0.389 MB ONNX; 24 h estimate = 17,280 epochs x 1.116 ms (ONNX CPU, model only).
- In the TrueSleep application the same model analyses a 24.4 h DMH recording end-to-end (feature extraction + inference + postprocessing) in 25.1 s.
- Legacy Hypnos V446/V448: parameter count was not recorded in the 2026-09 artefacts; weights are 0.24 MB each.
- LGBM rows store trees/arrays, not float parameters; the file size is the relevant memory figure.
- SleepTransformer / SleepGPT / DeepSleepNet 24 h timings use stride 1 and include context-window construction; external timings are not directly comparable across implementations.
