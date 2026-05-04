# `experiments/ziming/`

Snapshot of Ziming Li's task 3 + task 7 evaluation runs, May 2026.

## Scope

- **Tasks**: 3 (host-guest knowledge retrieval) and 7 (solvent
  classification, 6-way).
- **Models** (11): claude-sonnet-4.6, deepseek-v4-pro,
  gemini-3-flash-preview (high / nothinking), gpt-5.4-nano (xhigh /
  nothinking), gpt-5.5 (xhigh / nothinking), llama-3.1-8b-instruct,
  llama-3.1-70b-instruct, qwen3.5-9b, qwen3.5-27b.
- **Settings** (3): `base` / `fewshot` / `cot`.
- **Inference backend**: OpenRouter chat-completions API.
- **Total runs**: 11 × 3 × 2 = 66 (predictions/) sets.

## Layout

```
ziming/
├── task3/        # was "task1" in the upstream local working copy
│   ├── scripts/  # 7 scripts: build_prompts → run_inference → evaluate → plot → explore_metrics
│   ├── eval/     # per-row CSVs, leaderboard.csv, figures/
│   └── README.md
└── task7/        # was "task3" in the upstream local working copy
    ├── scripts/  # 6 scripts (data_prep → build_prompts → run_inference → evaluate → plot + relaunch helper)
    ├── eval/     # per-class CSVs, leaderboard.csv, figures/
    └── README.md
```

## Key takeaways

- **Task 3** — the conventional ROUGE-L F1 metric is **anti-scaling**
  on this task (Spearman with model size = −0.36). After applying a
  `parse_status != fallback_full` filter and switching to **ROUGE-1
  recall on the reverse subtype**, scaling is recovered (+0.89). See
  `task3/README.md` for the full story and `task3/eval/figures/` for
  the comparison plots.
- **Task 7** — standard macro-F1 ranking is reasonable; see
  `task7/eval/figures/fig_task7_macroF1.png`.

## Reproducibility caveat

Scripts under `*/scripts/` retain their original Anvil paths (e.g.
`outputs/task1/`, `outputs/task3/`) because they were lifted verbatim.
To rerun, set `ROOT` in each script (top of file: `ROOT = Path(...)`)
to a local working directory holding `data/task{3,7}/` with the
matching prompt files, then run `02 → 03 → 04 → 05` in order.
