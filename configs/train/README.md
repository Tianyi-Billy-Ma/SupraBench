# `configs/train/`

Training-time YAML configs. One file per (recipe, base-model, corpus)
combination. Loaded by `src/train/cpt_lora.py` (and future siblings).

| File | Recipe |
| --- | --- |
| `cpt_qwen35_eupmc.yaml` | LoRA continued pretraining of `Qwen/Qwen3.5-27B` on `mtybilly/EU-PMC` (filtered split). |

Conventions:

- Use full Hugging Face Hub ids (`Qwen/Qwen3.5-27B`, `mtybilly/EU-PMC`).
- Per-cluster paths (HF cache, output dir) come from environment
  variables set by `scripts/<cluster>/bashrc.sh`, never hardcoded here.
- CLI overrides use dotted keys, e.g.
  `--override training.max_steps=20 dataset.train_rows=256`.
