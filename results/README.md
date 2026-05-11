# SupraBench evaluation results

Captured eval metrics + analysis for Qwen3.5-27B (Base, Few-Shot, CoT), our
two CPT'd LoRA adapters (v1 / v2), and the inference-time guided-decoding
follow-up (v2_guided). Raw `metrics.json` files live under `metrics/`; raw
prediction `.jsonl` files for the three Qwen3.5-27B paper baselines live
under `results_task1_qwen3_5_27b/`. This README is the human-readable index.

```
results/
  metrics/                                 # one dir per (task, model)
    task<N>_base_<model>/metrics.json
  results_task1_qwen3_5_27b/               # paper-table baseline prediction dumps
    results_task1_v1_qwen3_5_27b.jsonl     # ≙ Base   (matches paper MAE 1.725)
    results_task1_v2_qwen3_5_27b.jsonl     # ≙ Few-Shot (paper MAE 2.237)
    results_task1_v3_qwen3_5_27b.jsonl     # ≙ CoT    (paper MAE 2.389)
  analysis/
    task1_outliers.md                      # robust-stats diagnosis
    task1_failure_examples.md              # representative v2 failure traces
  README.md                                # this file
```

## Models compared

| Tag         | Setup                                                                                       | Notes |
| ----------- | ------------------------------------------------------------------------------------------- | ----- |
| `base`      | Qwen3.5-27B straight off the Hub                                                            | Paper Table 1 row — derived from `results_task1_v1_qwen3_5_27b.jsonl` |
| `fewshot`   | Qwen3.5-27B + few-shot prompt                                                               | Paper Table 1 row — derived from `results_task1_v2_qwen3_5_27b.jsonl` |
| `cot`       | Qwen3.5-27B + CoT prompt                                                                    | Paper Table 1 row — derived from `results_task1_v3_qwen3_5_27b.jsonl` |
| `v1`        | LoRA r=64, lr=2e-5, 2 epochs on EU-PMC[filtered] (~75M tokens)                              | First CPT attempt |
| `v2`        | LoRA r=32, lr=1e-5, 1 epoch on EvoLM-style mix (~20M tokens, 80/15/5 split)                 | Second CPT attempt; recipe from EvoLM Table 2 |
| `v2_guided` | same v2 adapter, **inference-time** `<answer>\n` prompt-suffix + `</answer>` stop string    | Tests whether v2's failure is format-only or knowledge-also |

## Task 1 — logKa prediction (MAE ↓, lower is better)

### Headline metrics

| | n_parsed / total | MAE | RMSE | acc@0.5 | acc@1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **base**    | 2392/2392 | **1.725** | **2.496** | — | — |
| fewshot     | 2392/2392 | 2.237 | 3.388 | — | — |
| cot         | 2392/2392 | 2.389 | 3.504 | — | — |
| v1 (ours)   | 2242/2392 | 2.188 | 2.748 | 0.122 | 0.249 |
| v2 (ours)   | 2106/2392 | 2.266 | 2.889 | 0.118 | 0.253 |
| v2_guided   | 2250/2392 | 2.232 | 2.876 | 0.132 | 0.263 |

### Robust statistics

| | MAE | **MedAE** | MAE 5%-trim | MAE clipped [-2,15] |
| --- | ---: | ---: | ---: | ---: |
| **base**    | 1.725 | **1.220** | **1.452** | 1.676 |
| fewshot     | 2.237 | 1.326 | 1.850 | 1.702 |
| cot         | 2.389 | 1.444 | 2.008 | 1.811 |
| v1 (ours)   | 2.188 | 1.838 | 1.958 | 2.102 |
| v2 (ours)   | 2.266 | 1.893 | 2.015 | 2.167 |
| v2_guided   | 2.232 | 1.854 | 1.966 | 2.071 |

### Reading this table

**Base Qwen3.5-27B MedAE is 1.220 — *better* than its MAE of 1.725 lets on.**
Almost all of the gap between Base-MAE (1.725) and our v2 (2.27) holds up
under robust statistics:

- Base→v2 under MAE:   ∆ 0.541
- Base→v2 under MedAE: ∆ 0.673
- Base→v2 under 5%-trimmed: ∆ 0.563

The adapters degrade the *typical* prediction, not just the tail. The
"v2's chemistry is OK, only the format is broken" story is **wrong** under
the proper baseline comparison; the LoRA adapters actively damage point
estimates across the distribution. Guided decoding closes a small fraction
of the gap (MedAE 1.893 → 1.854) but cannot recover the bulk.

A separate observation: the paper's Few-Shot and CoT MAEs are heavily
tail-driven (preds up to 41, hundreds of negative outliers). Under MedAE
they're much closer to Base (1.326 / 1.444 vs Base's 1.220). The paper's
headline MAE understates how well Qwen3.5-27B actually does under all
three prompting strategies.

## Task 2 — host classification (ACC ↑)

| | n_parsed / total | ACC |
| --- | ---: | ---: |
| base (paper) | — | **0.402** |
| v1 (ours)    | 2064/2064 | 0.261 |
| v2 (ours)    | 2064/2064 | 0.260 |

Both adapters mode-collapse onto a single class. Identical to three
decimal places.

## Task 3 — host description (Rouge-L ↑)

| | n_parsed / total | rougeL_f | kh |
| --- | ---: | ---: | ---: |
| base (paper) | — | **0.310** | — |
| v1 (ours)    | crash | n/a   | n/a (rouge-score not installed at the time) |
| v2 (ours)    | 122/122 | 0.011 | 0.000 |

v2 collapse is catastrophic — output is pure internal monologue ("Here's a
thinking process that leads to the suggested answer: …"), zero overlap
with reference descriptions.

## Task 7 — solvent classification (macro-F1 ↑)

| | n_parsed / total | accuracy | macro_f1 | weighted_f1 | parse_fail_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| base (paper) | — | — | **0.230** | — | — |
| v1 (ours)    | 1955/1955 | 0.930 | 0.161 | 0.897 | 0.000 |
| v2 (ours)    | 1955/1955 | 0.930 | 0.161 | 0.897 | 0.000 |

Both adapters predict "water" (93%-majority class) for every example. The
macro-F1 = 0.16066066… is bit-identical between v1 and v2 because both
collapse identically on the same downstream label space, despite differing
predictions verbatim (v1↔v2 prediction file diff is 18.6 KB).

## Headline verdict (revised)

Neither v1 nor v2 beats `base` (or even `cot` / `fewshot`) on any task.
The regression is robust to outlier filtering and choice of metric — it
shows up under MAE, MedAE, trimmed mean, and clipped-range MAE alike.

The failure mode is **format-collapse plus knowledge degradation**: the
adapter is over-trained on raw academic-abstract prose, which both
(a) suppresses the structured `<answer>…</answer>` output and (b)
shifts the model's point estimates away from chemistry-consistent
values. Inference-time guidance (v2_guided) fixes some of (a) but
cannot recover (b).

## Recommended next steps

1. **Tighten the parser clip range** from `[-10, 30]` to `[-2, 15]` and
   add `medae` + `mae_5pct_trimmed` to the standard task-1 metrics dict
   so the paper can report robust stats alongside MAE.
2. **v3 reading-comprehension DAPT** — convert the supramolecular
   subset into Q&A pairs that train on `<answer>…</answer>` outputs
   directly, so the gradient signal teaches knowledge *and* format
   simultaneously rather than letting raw-prose continuation amplify
   the base model's RLHF-contamination monologue tendency.
3. **Run our adapters with `cot` and `fewshot` prompting too** — we
   only have `base` prompts for v1 / v2. If our adapters interact
   differently with explicit reasoning prompts, that's a free data
   point for the diagnosis.
