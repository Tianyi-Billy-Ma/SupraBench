# Task 1 outlier-and-robust-stats diagnosis

This note replaces the earlier version once the three Qwen3.5-27B baseline
prediction files (Base / Few-Shot / CoT) became available under
`../results_task1_qwen3_5_27b/`. The earlier note speculated that the
v1→v2 regression was tail-driven; with the actual base predictions in
hand, that hypothesis turns out to be **wrong**.

## Reference distribution

```
n=2392   min=0.00   max=12.04   median=3.57   p99=10.81
refs in [0,12]: 2391 (100.0%)   refs > 15: 0   refs < 0: 0
```

A prediction outside `[-2, 15]` is chemically implausible — every reference
except a single 0.00 lies in `[0, 12]`.

## Current parser filter is loose

`src/eval/task1.py::_coerce_logka` drops only values `< -10` or `> 30`.
Everything in `[-10, 30]` is accepted into the MAE sum.

## Prediction distributions

| | range | preds > 15 | preds < 0 | worst single \|err\| |
| --- | --- | ---: | ---: | ---: |
| **base**    | [-10, 41] | 2 (0.08%)   | 43 (1.80%)   | **37.04** |
| fewshot     | [-10, 41] | 29 (1.21%)  | 306 (12.79%) | 37.06 |
| cot         | [-6, 39]  | 22 (0.92%)  | **374 (15.64%)** | 37.06 |
| v1 (ours)   | [-10, 22] | 3 (0.13%)   | 40 (1.78%)   | 18.4 |
| v2 (ours)   | [-10, 24] | 3 (0.14%)   | 37 (1.76%)   | 23.1 |
| v2_guided   | [-10, 18] | 2 (0.09%)   | **74 (3.29%)** | 16.2 |

Three observations:

- **Base is by far the best-calibrated**: 2 high-side outliers, only 43
  negative preds.
- **Few-Shot and CoT are extremely tail-heavy** on the negative side
  (12–16 % of preds < 0). This is what drives their MAEs *up* relative to
  Base under the headline metric.
- **Our adapters compress the prediction range** (max 22–24 vs Base's 41)
  and have fewer outliers in absolute count — but their *typical*
  predictions are systematically worse. See robust stats below.

## Robust summary statistics

| | MAE | **MedAE** | MAE 5%-trim | MAE clipped [-2,15] |
| --- | ---: | ---: | ---: | ---: |
| **base**    | 1.725 | **1.220** | **1.452** | 1.676 |
| fewshot     | 2.237 | 1.326 | 1.850 | 1.702 |
| cot         | 2.389 | 1.444 | 2.008 | 1.811 |
| v1 (ours)   | 2.188 | 1.838 | 1.958 | 2.102 |
| v2 (ours)   | 2.266 | 1.893 | 2.015 | 2.167 |
| v2_guided   | 2.232 | 1.854 | 1.966 | 2.071 |

The Base-vs-CPT-adapter gap *grows* slightly under MedAE (∆ 0.61–0.67)
relative to MAE (∆ 0.46–0.54). Our adapters underperform Base on the
*typical* prediction, not just the tail.

The Few-Shot-vs-CoT-vs-Base internal gap *shrinks* under MedAE
(Base 1.220 / FS 1.326 / CoT 1.444; spread 0.22) relative to MAE
(Base 1.725 / FS 2.237 / CoT 2.389; spread 0.66). The Few-Shot and CoT
headline MAEs are inflated by their many extreme outliers (≤-10, ≥30).

## Top-10 absolute errors

| | top-10 abs errors |
| --- | --- |
| base       | 37.04, 15.09, 12.90, 11.56, 10.63,  9.77,  9.60,  9.49,  8.92,  8.89 |
| fewshot    | 37.06, 37.04, 15.90, 12.25, 12.01, 11.50, 11.42, 11.32, 11.14, 11.11 |
| cot        | 37.06, 16.80, 13.54, 12.70, 12.52, 12.01, 11.92, 11.61, 11.30, 11.28 |
| v1 (ours)  | 18.38, 16.20, 15.94, 12.75, 12.69, 11.00, 10.70,  9.12,  9.00,  8.00 |
| v2 (ours)  | 23.05, 16.20, 15.99, 15.94, 13.81, 13.42, 12.75, 12.69, 11.07, 11.00 |
| v2_guided  | 16.20, 15.99, 15.94, 13.81, 13.42, 12.75, 12.69, 12.15, 11.94, 11.90 |

Even Base has one 37-logKa-off prediction (a presumed Ka→logKa miscoding
inside `_coerce_logka`). Tightening the parser would help all six variants
slightly, but doesn't change the verdict.

## Recommendations

1. Tighten the parser clip range from `[-10, 30]` to `[-2, 15]`. Treat
   chemistry-implausible predictions as parse failures.
2. Add `medae` and `mae_5pct_trimmed` to `Task1Evaluator`'s output. Report
   all four (MAE / RMSE / MedAE / trimmed) in the paper.
3. The v1→v2 regression is **not** primarily an outlier effect; v3 (or
   other interventions targeting knowledge, not just format) is justified.
