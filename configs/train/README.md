# `configs/train/`

Training-time YAML configs. One file per (recipe, base-model, corpus)
combination. Loaded by `src/train/cpt_lora.py` (and future siblings).

| File                            | Recipe                                                                                       |
| ------------------------------- | -------------------------------------------------------------------------------------------- |
| `cpt_qwen35_eupmc.yaml` (v1)    | LoRA CPT of `Qwen/Qwen3.5-27B` on `mtybilly/EU-PMC[filtered]` — broad-domain, regressed.     |
| `cpt_qwen35_supra_v2.yaml` (v2) | LoRA CPT on `mtybilly/SupraBench-CPT-Mix-v2` (80 % supra-EU-PMC + 15 % FineWeb-Edu + 5 % Tulu-3 anchor). |

## v2 design

The v1 LoRA catastrophically forgot instruction-following: Task 7 macro-F1
fell from 0.230 → 0.161 (model collapsed onto the majority class), Task 2
ACC from 0.402 → 0.261, Task 1 MAE 1.73 → 2.19. v2 is a **data-mix
re-design** rather than a fresh pretrain.

**Recipe** — informed by EvoLM (Qi et al. 2025, [arxiv 2506.16029](https://arxiv.org/abs/2506.16029)),
whose Table 2 shows replay ratio ≈5–16 % is the empirical sweet spot.

| Stream         | Source                                       | Target % of tokens |
| -------------- | -------------------------------------------- | ------------------ |
| domain         | `mtybilly/EU-PMC[supramolecular]`            | 80                 |
| replay         | `HuggingFaceFW/fineweb-edu` (sample-10BT)    | 15                 |
| format-anchor  | `allenai/tulu-3-sft-mixture` (raw-flattened) | 5                  |

The 5 % format-anchor stream is our deliberate addition (not in EvoLM)
to specifically counter the v1 Task 7 mode-collapse. Tulu-3 turns are
flattened into raw `Question:\n... Answer:\n...` text — no chat template,
so the run remains a pure raw-text CPT.

**Domain-corpus tightening.** v1 used `mtybilly/EU-PMC[filtered]` (134K,
≈37M tok), which still admits broad polymer / MOF / coordination
chemistry. v2 applies a strict named-host-or-binding-term regex (see
`scripts/data/build_supra_subset.py`) that keeps **49,214 articles**
(36.8 %, ≈17M tok).

**Hyperparameter tightening.** Halve LoRA rank (64→32), halve LR
(2e-5→1e-5), single epoch, longer warmup (3→5 %). EvoLM warns that
over-training a small adapter on a smaller corpus is the dominant
forgetting failure mode.

## Building / rebuilding the corpora

```
# 1. Filter EU-PMC → push as mtybilly/EU-PMC config=supramolecular
uv run --extra hf python scripts/data/build_supra_subset.py
#    (use --dry-run to preview counts before pushing)

# 2. Build the v2 mix → push as mtybilly/SupraBench-CPT-Mix-v2
uv run --extra hf python scripts/data/build_cpt_mix.py
```

## Conventions

- Use full Hugging Face Hub ids (`Qwen/Qwen3.5-27B`, `mtybilly/EU-PMC`).
- Per-cluster paths (HF cache, output dir) come from environment
  variables set by `scripts/<cluster>/bashrc.sh`, never hardcoded here.
- CLI overrides use dotted keys, e.g.
  `--override training.max_steps=20 dataset.train_rows=256`.
