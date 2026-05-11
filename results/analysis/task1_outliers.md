# Task 1 outlier-and-robust-stats diagnosis

The headline MAE values in `../README.md` mix in tail behavior. This note shows
where the tail lives and how robust statistics shift the picture.

## Reference distribution

```
n=2392   min=0.00   max=12.04   median=3.57   p99=10.81
refs in [0,12]: 2391 (100.0%)   refs > 15: 0   refs < 0: 0
```

Every reference except a single 0.00 lies in `[0, 12]`. A prediction outside
`[-2, 15]` is chemically implausible.

## Current parser filter is loose

`src/eval/task1.py::_coerce_logka` drops only values `< -10` or `> 30`. Anything
in `[-10, 30]` is accepted into the MAE sum.

## Prediction distributions

| | range | preds > 15 | preds < 0 | worst single \|err\| |
| --- | --- | ---: | ---: | ---: |
| v1        | [-10, 22] | 3 (0.13%)  | 40 (1.78%)  | **18.4** |
| v2        | [-10, 24] | 3 (0.14%)  | 37 (1.76%)  | **23.1** |
| v2_guided | [-10, 18] | 2 (0.09%)  | **74 (3.29%)** | 16.2 |

v2_guided traded a few high-side outliers for many more `0`/`-10` lower-tail
spikes — the model under guidance commits to a number fast, including when it
has no idea.

## Robust summary stats

| | MAE | MedAE | MAE 5%-trim | MAE clipped [0,12] |
| --- | ---: | ---: | ---: | ---: |
| v1        | 2.188 | **1.838** | 1.958 | 2.102 |
| v2        | 2.266 | 1.893 | 2.015 | 2.167 |
| v2_guided | 2.232 | 1.854 | 1.966 | 2.071 |

The all-parsed MAE spread between v1, v2, v2_guided is **0.078**.
The MedAE spread is **0.055**.

About half of the apparent v1→v2 regression is tail behavior, not core
distributional shift. v2_guided's MedAE is statistically within noise of v1's.

## Top-10 absolute errors

| | top-10 abs errors |
| --- | --- |
| v1        | 18.38, 16.20, 15.94, 12.75, 12.69, 11.00, 10.70, 9.12, 9.00, 8.00 |
| v2        | 23.05, 16.20, 15.99, 15.94, 13.81, 13.42, 12.75, 12.69, 11.07, 11.00 |
| v2_guided | 16.20, 15.99, 15.94, 13.81, 13.42, 12.75, 12.69, 12.15, 11.94, 11.90 |

A single 23-logKa-off prediction in v2 contributes 0.011 to MAE on its own.
The top-10 outliers together contribute ~0.06 to MAE.

## Recommendation

1. Tighten parser clip range from `[-10, 30]` to `[-2, 15]`. Treat
   chemistry-implausible predictions as parse failures, not 20-logKa errors.
2. Add `MedAE` (median absolute error) to `Task1Evaluator`'s output dict
   alongside `mae` / `rmse`. Report both in the paper.
3. Run the un-adapted base Qwen3.5-27B through the same eval to capture its
   robust stats for an apples-to-apples comparison — the paper-reported MAE
   of 1.725 says nothing about the base model's tail behavior.
