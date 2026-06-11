# SupraBench evaluation results

Raw per-model predictions and the metrics computed from them, for the four
scored SupraBench tasks (`bap`, `tbs`, `hgd`, `sid`) across the `base`,
`fewshot`, and `cot` prompting strategies.

```
results/
  <task>/<strategy>/<model>.jsonl      # one record per example: prompt, reference, prediction, raw response
  <task>/<strategy>/<model>.meta.json  # per-run metadata (where present)
  metrics/<run>/metrics.json           # metrics for one evaluated run, produced by src/eval/
```

- `<task>` ∈ `{bap, tbs, hgd, sid}` — see the top-level `README.md` for the task definitions.
- `<strategy>` ∈ `{base, fewshot, cot}`.

Metric definitions live in `src/eval/metrics/`; the cross-model summary table
can be regenerated with `scripts/aggregate_results.py`.
