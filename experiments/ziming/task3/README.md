# `experiments/ziming/task3/`

Host-guest binding knowledge retrieval. Two subtypes per row:

- **forward** (35 rows): given a macrocyclic host, predict its
  high-affinity guests + aggregate properties (MW, charge, ring count,
  H-bond donors/acceptors, logKa threshold).
- **reverse** (87 rows): given a guest molecule + SMILES, predict the
  hosts that bind it strongly with their **logKa values** + host-family
  description.

Total: 122 examples per (model, setting). Raw data shipped under
`../../../data/task3/` (`task3.jsonl` + `prompts_{base,fewshot,cot}.jsonl`).

## Layout

```
task3/
├── scripts/
│   ├── 02_build_prompts.py        # 02c_build_prompts_task1.py
│   ├── 03_run_inference.py        # 03c_run_inference_task1.py
│   ├── 03d_inference_openrouter.py
│   ├── 04_evaluate.py             # produces leaderboard.csv + per_row CSVs
│   ├── 05_plot.py                 # produces 4 figures
│   ├── 06_explore_metrics.py      # v1 metric scan
│   └── 06b_explore_more.py        # v2: + ROUGE-2, METEOR, chrF, IDF-weighted, TF-IDF cosine
└── eval/
    ├── leaderboard.csv
    ├── <model>_<setting>_per_row.csv     # one row per example
    ├── metric_explore_per_row.csv        # output of 06_
    ├── metric_explore_summary.csv
    ├── metric_scaling_corr.csv           # Spearman vs log10(size)
    ├── metric_explore_v2_*.csv           # output of 06b_
    └── figures/
        ├── fig_task3_rougeL.png/pdf      # ROUGE-L F1 (anti-scaling, conventional metric)
        ├── fig_task3_kh.png/pdf          # Keyword Hit recall
        ├── fig_task3_rouge1r.png/pdf     # ROUGE-1 recall, all rows
        └── fig_task3_rouge1r_rev.png/pdf # ROUGE-1 recall, reverse subtype only (best scaling)
```

## Metrics in `leaderboard.csv`

| column | meaning | scaling? |
| --- | --- | --- |
| `n` / `n_valid` / `parse_fail_rate` | quality filter (drop `parse_status='fallback_full'`) | — |
| `rougeL_all` / `rougeL_fwd` / `rougeL_rev` | LCS-based F1, mean over valid rows | Spearman = −0.36 |
| `rouge1r_all` / `rouge1r_fwd` / `rouge1r_rev` | ROUGE-1 recall (unigram, no order) | rev = **+0.89** |
| `kh_all` / `kh_fwd` / `kh_rev` | substring recall over gold compound names | +0.25 |

Spearman is computed against `log10(model_size_B)` over 7 models
(claude, deepseek, gemini-flash_nothinking, llama-70b, llama-8b, qwen-27b,
qwen-9b).

## Why `rouge1r_rev` is the recommended scaling metric

- **F1 is anti-scaling** on this task: frontier models (Claude,
  DeepSeek) paraphrase gold's content with their own chemistry
  vocabulary → low precision → low F1. Small models that copy the
  fewshot template verbatim get high precision and rank #1.
- **Recall-only** (`rouge1_r`) drops the precision penalty, so
  paraphrase no longer hurts.
- **Reverse subtype** filters out the part of the task where small
  models cheat by copying templates: forward gold has a fixed
  "Representative guests:" framing that any model can mimic; reverse
  gold has structured `host_name (logKa=N)` entries unique per row,
  which actually require chemistry knowledge to fill in.
- Result: clean ranking deepseek > claude > gemini > llama-70b >
  llama-8b ≈ qwen-27b > qwen-9b.

## Why qwen3.5-9b drops to ~0

97% of its outputs are unparsable reasoning-trace leak (~13k chars of
"Let's refine the MW… Let's check…" with no final answer). The
`parse_status` filter retains only 4 valid rows, all of which still
contain noise. This is a model failure, not a metric failure.
