# `experiments/ziming/task7/`

Solvent classification: given a host-guest binding interaction, predict
the solvent in which the binding affinity (logKa) was measured. 6-way
classification:

```
water, DMSO, MeCN, MeOH, CHCl3, CH2Cl2
```

The label distribution is heavily skewed toward `water` (≈70% of all
examples). Raw data shipped under `../../../data/task7/` (`eval.parquet`
+ `prompts_{base,fewshot,cot}.jsonl` + `host_smiles_dict.json`).

## Layout

```
task7/
├── scripts/
│   ├── 01_data_prep.py         # 01b_data_prep_task3.py
│   ├── 02_build_prompts.py     # 02b_build_prompts_task3.py
│   ├── 03_run_inference.py     # 03d_run_inference_openrouter.py
│   ├── 04_evaluate.py          # produces leaderboard.csv + per_class CSVs
│   ├── 05_plot.py              # produces 1 figure (macro-F1 grouped bar)
│   └── relaunch_incomplete.py  # resume helper for partial runs
└── eval/
    ├── leaderboard.csv
    ├── <model>_<setting>_per_class.csv
    └── figures/
        └── fig_task7_macroF1.png/pdf
```

## Metrics in `leaderboard.csv`

| column | meaning |
| --- | --- |
| `n` / `parse_fail` / `parse_fail_rate` | dedupe by `interaction_id`, count unparseable predictions |
| `accuracy` | exact-match accuracy on parseable rows |
| `macro_f1` | unweighted mean of per-class F1 across **classes with support > 0** |
| `weighted_f1` | per-class F1 weighted by support |

Per-class precision / recall / F1 / support are reported per
`<model>_<setting>_per_class.csv`.

## Headline observations (fewshot)

- **gemini-3-flash-preview_nothinking** leads (macro-F1 ≈ 0.39);
  cot/base only marginally lower.
- **deepseek-v4-pro** is one of the few models where `cot` improves
  over `base` (0.31 → 0.34).
- **claude-sonnet-4.6** is prompt-insensitive (~0.21 across all three
  settings).
- **qwen3.5-27b** prefers `cot` over `fewshot` (0.26 vs 0.20).
- **llama-3.1-70b-instruct** has its worst score under `cot` (0.11).

Figure `fig_task7_macroF1.png` shows all 10 retained models × 3 settings
(excludes gpt-5.5 variants and gemini-flash-preview_high, which had
pervasive parsing failures).

`gpt-5.4-mini_nothinking` (added 2026-05) prefers `cot`/`base`
(macro-F1 ≈ 0.22–0.23) over `fewshot` (0.155); the long fewshot prompt
triggers ~11% parse failures that drag the metric down.
