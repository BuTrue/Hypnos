# Statistical significance (paired bootstrap)

Protocol: 31 recordings, each scored once out-of-fold. Difference = Hypnos minus
SleepGPT per recording; 20,000 paired bootstrap resamples (seed 0). This tests
whether the ranking is stable given only 31 mice/recordings.

| comparison | metric | mean difference | 95% CI | P(Hypnos > SleepGPT) |
|---|---|---|---|---|
| Hypnos-V446 - SleepGPT | REM F1 | **+0.0124** | [-0.0586, +0.0777] | 0.650 |
| Hypnos-V446 - SleepGPT | mean F1 | **+0.0147** | [-0.0155, +0.0470] | 0.824 |
| Hypnos CV5 10EP - SleepGPT | REM F1 | -0.0428 | [-0.1072, +0.0233] | 0.099 |
| Hypnos CV5 10EP - SleepGPT | mean F1 | -0.0147 | [-0.0455, +0.0192] | 0.186 |

## Reading

- **V446 vs SleepGPT is statistically indistinguishable.** The point estimate
  favours V446, but the 95% CI spans zero and P = 0.65. With 31 recordings we
  cannot claim a win.
- **CV5 10EP vs SleepGPT trends lower but is not significant at 0.05**
  (P = 0.099 for REM). The point estimate is -0.043 REM; the CI includes zero.
- The small sample is the limiting factor. A 0.01–0.04 F1 difference is within
  the noise of 31 recordings and a handful of REM-poor recordings.
- SleepGPT here is an in-house PyTorch port trained from scratch on the same
  folds, not the authors' released weights. Port fidelity alone can move F1 by
  a few points.
- Input representation and capacity differ by design: SleepGPT has 1.67 M
  parameters and a 9×20 time–frequency context; CV5 has 88.6 k parameters and a
  184×22 spectrogram context. This is a comparison of methods, not a
  controlled capacity match.

## Efficiency context

| model | params | weights | 24 h inference | REM F1 |
|---|---|---|---|---|
| Hypnos-V446 | — | 0.24 MB | 6.3 s | 0.7108 |
| Hypnos CV5 10EP | 88,607 | 0.37 MB | 19.3 s | 0.6535 |
| SleepGPT | 1,670,371 | 6.72 MB | 73.4 s | 0.6994 |

V446 is roughly 28× smaller and 12× faster on a 24 h file than SleepGPT, with a
statistically tied REM F1. CV5 is 17× smaller and 3.8× faster, with a lower but
non-significant point estimate.

## Reproduce

```python
# per-recording OOF F1, then paired bootstrap over recordings
d = np.array([f1_hypnos[r] - f1_sleepgpt[r] for r in common])
bs = [np.mean(np.random.choice(d, len(d), replace=True)) for _ in range(20000)]
```
