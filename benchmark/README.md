# Benchmark assets

| path | content |
|---|---|
| `results/benchmark_master.md` | human-readable main table (14 models) |
| `results/benchmark_master.csv` | machine-readable main table |
| `results/benchmark_master.json` | full metadata (per-fold std, params, source) |
| `results/resources.md` | parameters, size, memory, latency, training and 24 h inference time |
| `results/hmm_comparison.md` | raw vs HMM postprocessing, all models |
| `results/cv5_10ep_eval.md` | Hypnos 10EP raw/HMM variants |
| `results/cv5_30ep_eval.md` | Hypnos 30EP pilot raw/HMM variants |
| `results/epi9_eval.md` | Hypnos EPI9 raw/HMM variants |
| `results/per_fold.csv` | Hypnos CV5 per-fold REM/Wake/NREM and training time |
| `figures/f1_by_class.png` | per-class F1 for every model |
| `figures/benchmark_summary.png` | sum F1, speed vs size, training cost, static memory |
| `code/make_charts.py` | regenerate both figures from `benchmark_master.json` |
| `code/aggregate_external.py` | aggregation logic used for external result JSONs |

## Regenerate figures

```bash
python benchmark/code/make_charts.py
```

## Rebuild the table

The master table is assembled from one aggregate JSON per external model plus
the Hypnos fold files. Those raw per-model files are not included because they
contain absolute paths from the private training machine. The aggregation rule
is in `code/aggregate_external.py`; the schema is:

```json
{
  "fold0": {
    "test_per_rec": {"rec": {"f1_rem": 0.0, "f1_wake": 0.0, "f1_nrem": 0.0}},
    "train_s": 0.0,
    "n_params": 0
  }
}
```

Per fold: average each class F1 over that fold's test recordings. Final:
mean over five folds; `std REM` is the population standard deviation of the
five fold means. No postprocessing is applied.

## Protocol

See [`../docs/BENCHMARK.md`](../docs/BENCHMARK.md) for folds, scoring and
limitations.
