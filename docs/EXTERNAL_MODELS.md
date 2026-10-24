# External models

The benchmark compares Hypnos against models published for sleep staging. All
external models were **re-implemented in PyTorch for this corpus** and trained
from scratch on the same five folds. No external code or weights are
redistributed in this repository.

## Benchmarked models

| model | paper | repository | license | input used here | port notes |
|---|---|---|---|---|---|
| AccuSleep | Barger et al., 2019 | [zekebarger/AccuSleep](https://github.com/zekebarger/AccuSleep) | GPL-3.0 | 5 s spectrogram | compact CNN; original MATLAB pipeline, Python-style port |
| AttnSleep | Eldele et al., IEEE TNNLS 2021 | [emadeldeen24/AttnSleep](https://github.com/emadeldeen24/AttnSleep) | MIT | 5 s spectrogram | SE-CNN + temporal context encoder; REM collapse on mouse data |
| DeepSleepNet | Supratak et al., IEEE TNSRE 2017 | [akaraspt/deepsleepnet](https://github.com/akaraspt/deepsleepnet) | Apache-2.0 | raw EEG 5 s, 20-epoch sequences | two CNN streams + BiLSTM + residual; 7.6 M params |
| SleepTransformer | Phan et al., IEEE TBME 2022 | [pquochuy/SleepTransformer](https://github.com/pquochuy/SleepTransformer) | MIT | 5 s spectrogram, 9×20 context | two-level Transformer; spectrogram cache reused across windows |
| SleepGPT | — | [LordXX505/SleepGPT](https://github.com/LordXX505/SleepGPT) | no license file | 5 s time–frequency, 9×20 | dual time/frequency Transformer; 184→23 frequency pooling |
| TinySleepNet | Supratak & Guo, IEEE Access 2020 | [akaraspt/tinysleepnet](https://github.com/akaraspt/tinysleepnet) | Apache-2.0 | raw EEG 5 s, 20-epoch sequences | CNN + GRU; single-epoch output, no sequence smoothing |
| U-Sleep (lightweight) | Perslev et al., npj Digital Medicine 2021 | [perslev/u-time](https://github.com/perslev/u-time) | MIT | raw EEG 5 s, 20-epoch sequences | lightweight U-Net port (38 k params), not the full 1.0 M model |
| LGBM-153feat | IntelliSleepScorer feature set | [broadinstitute/IntelliSleepScorer](https://github.com/broadinstitute/IntelliSleepScorer) | no license file | 153 hand-crafted features / 5 s | LightGBM on spectral + temporal features |
| LGBM-G2 | YASA transfer | [raphaelvallat/yasa](https://github.com/raphaelvallat/yasa) | BSD-3-Clause | 153 features, YASA-pretrained trunk | human-pretrained feature extractor transferred to mouse data |
| YASA (raw) | Vallat & Walker, JOSS 2021 | [raphaelvallat/yasa](https://github.com/raphaelvallat/yasa) | BSD-3-Clause | human EEG/EMG | evaluated as-is on 2 recordings only; almost all NREM (domain gap), excluded from the main table |

`SeqSleepNet` ([Kennethborup/SeqSleepNet](https://github.com/Kennethborup/SeqSleepNet))
and the full U-Time model were cloned for reference but not included in the
final table.

## Why these models

They cover the main design families in rodent and human sleep staging:

- **compact CNNs** (AccuSleep, TinySleepNet) — cheap, strong on Wake/NREM,
  weak on REM in mice;
- **sequence models** (DeepSleepNet, TinySleepNet) — raw EEG, recurrent
  context;
- **Transformers** (SleepTransformer, SleepGPT, AttnSleep) — long context,
  attention over time/frequency;
- **feature-based learners** (LGBM, LGBM-G2) — non-neural baselines with
  hand-crafted features and human transfer;
- **U-Net** (U-Sleep) — dense segmentation-style output.

## Fairness notes

- All models use the same 5 folds, the same training recordings per fold, the
  same 5 s labels and the same test recordings.
- Input representation differs by design (raw EEG vs spectrogram vs features).
  This is part of each model's published method; forcing a single input would
  not be a faithful comparison.
- Context lengths differ (single epoch, 9, 20, 22). The table reports each
  model's native configuration, not a matched-context ablation.
- External models were trained with a common budget policy, not tuned
  per model. Their published hyperparameters were used where available.
- Ports are not the authors' original TensorFlow/MATLAB code. Small
  implementation differences can change F1 by a few points.

## Licensing

Only aggregate benchmark numbers are published here. The external
repositories above retain their original licenses; in particular AccuSleep is
GPL-3.0 and SleepTransformer/TinySleepNet/U-Time have their own terms. If you
reuse an external architecture, follow its license and cite its paper. No
external weights are included in `models/`.
