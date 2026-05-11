# SupraBench evaluation results

Captured eval metrics + analysis for Qwen3.5-27B base, two CPT'd LoRA adapters,
and the guided-decoding follow-up experiment. Raw `metrics.json` files live
under `metrics/`; this README is the human-readable index.

Layout:

```
results/
  metrics/
    task<N>_base_<model>/metrics.json    # one per (task, model)
  analysis/
    task1_outliers.md                    # robust-stats diagnosis
    task1_failure_examples.md            # representative failure traces
  README.md                              # this file
```

## Models compared

| Tag         | Adapter                                                                                     | Notes |
| ----------- | ------------------------------------------------------------------------------------------- | ----- |
| `base`      | none — Qwen3.5-27B straight off the Hub                                                     | Paper baseline; we have only the headline MAE/ACC numbers, not predictions.jsonl |
| `v1`        | `outputs/cpt_qwen35_eupmc/adapter` (LoRA r=64, lr=2e-5, 2 epochs on EU-PMC[filtered], ~75M tokens) | First CPT attempt |
| `v2`        | `outputs/cpt_qwen35_supra_v2/adapter` (LoRA r=32, lr=1e-5, 1 epoch on the EvoLM-style mix, ~20M tokens) | Second CPT attempt; recipe from EvoLM Table 2 |
| `v2_guided` | same v2 adapter, **inference-time** prompt-suffix + stop-string output guidance              | Tests whether v2's failure is format-only or knowledge-also |

## Task 1 — logKa prediction (MAE ↓, lower is better)

| | n_parsed / total | MAE | RMSE | acc@0.5 | acc@1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base (paper)  | —          | **1.725** | **2.496** | — | — |
| v1            | 2242/2392  | 2.188     | 2.748     | 0.122 | 0.249 |
| v2            | 2106/2392  | 2.266     | 2.889     | 0.118 | 0.253 |
| v2_guided     | 2250/2392  | 2.232     | 2.876     | 0.132 | 0.263 |

**Robust stats** (computed offline from `predictions.jsonl`; see `analysis/task1_outliers.md`):

| | MAE | MedAE | MAE 5%-trim | MAE clipped to [0,12] |
| --- | ---: | ---: | ---: | ---: |
| v1        | 2.188 | 1.838 | 1.958 | 2.102 |
| v2        | 2.266 | 1.893 | 2.015 | 2.167 |
| v2_guided | 2.232 | 1.854 | 1.966 | 2.071 |

Under robust stats the three CPT variants converge — the headline MAE gap is
largely tail-driven by a handful of preds outside the plausible logKa range
(reference span = [0.00, 12.04]; parser currently clips only at [-10, 30]).

## Task 2 — host classification (ACC ↑)

| | n_parsed / total | ACC |
| --- | ---: | ---: |
| base (paper) | — | **0.402** |
| v1           | 2064/2064 | 0.261 |
| v2           | 2064/2064 | 0.260 |

Both adapters mode-collapse toward a single class. Identical performance to
3-decimal precision.

## Task 3 — host description (Rouge-L ↑)

| | n_parsed / total | rougeL_f | kh |
| --- | ---: | ---: | ---: |
| base (paper) | — | **0.310** | — |
| v1           | crash | n/a    | n/a (eval crashed; rouge-score not installed at the time) |
| v2           | 122/122 | 0.011 | 0.000 |

v2 collapse is catastrophic on this task — model emits pure internal monologue
("Here's a thinking process that leads to the suggested answer: …"), zero
overlap with reference descriptions.

## Task 7 — solvent classification (macro-F1 ↑)

| | n_parsed / total | accuracy | macro_f1 | weighted_f1 | parse_fail_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| base (paper) | — | — | **0.230** | — | — |
| v1           | 1955/1955 | 0.930 | 0.161 | 0.897 | 0.000 |
| v2           | 1955/1955 | 0.930 | 0.161 | 0.897 | 0.000 |

Both adapters predict "water" (the 93%-majority class) for every example. The
macro-F1 of 0.16066066… is bit-identical between v1 and v2 because both
trivially collapse on the same label distribution. The two adapters'
prediction files diverge by 18.6 KB (~thousand-byte diff in token-stream
verbiage before the final "water"), so they *are* different models — just
mode-collapsed onto the same downstream output.

## Headline verdict

Neither v1 nor v2 beats the base across any task. Both regressed on every
metric. The failure mode is **format / instruction-following collapse**, not
loss of chemistry knowledge per se — the v2 internal-monologue traces often
contain qualitatively correct chemistry reasoning that never gets compressed
into the required output shape.

The inference-time guided-decoding follow-up (`v2_guided`) showed that
prefix-injection alone (`<answer>\n` prepended, `</answer>` as stop string)
moves the headline MAE only 2.27 → 2.23 — the model continued its monologue
*inside* the `<answer>` tag. A stronger logits-level grammar constraint would
isolate the question more decisively, but the preponderance of evidence
already favors the "format-collapse, knowledge mostly intact" diagnosis.

## Next steps (recorded for the paper / next conversation)

1. Tighten the task-1 parser clip range from `[-10, 30]` to `[-2, 15]`
   (margin around plausible logKa) and add MedAE to the standard metrics so
   tail behavior stops dominating the headline.
2. Decide between (a) hard-grammar-constrained re-eval of v2 vs (b) v3 with
   reading-comprehension DAPT. See conversation notes.
3. Run the un-tuned base Qwen3.5-27B through our eval pipeline so we can
   compute *its* MedAE / robust stats — currently we only have the paper's
   reported MAE/ACC and can't apples-to-apples-compare under robust metrics.
