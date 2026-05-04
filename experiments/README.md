# `experiments/`

Self-contained snapshots of work done **outside** the main `src/` pipeline.

Each subdirectory is one contributor's deliverable: the dataset they
worked with, the scripts they ran, and the analysis they produced.
Snapshots are kept verbatim so reviewers can audit what was actually
executed; they are **not** wired into `src/main.py` and should not be
treated as the canonical implementation. Promotion of a snapshot into
the main pipeline (`src/datasets/<task>.py` + `src/eval/<task>.py` +
`configs/tasks/<task>.yaml`) is a separate, follow-on PR.

```
experiments/
└── <contributor>/
    └── <task>/
        ├── scripts/      # actual .py files that produced the results
        ├── eval/         # per-row CSVs, leaderboard, figures
        └── README.md
```

Raw model outputs (`predictions.csv`, `full_log.jsonl`) are intentionally
excluded — they are large and reproducible from `scripts/` + the
corresponding `data/<task>/` files. Contact the contributor if you need
the originals.
