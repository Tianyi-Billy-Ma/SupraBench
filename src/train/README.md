# `src/train/`

Placeholder for training / fine-tuning pipelines (SFT, preference tuning,
RL). Intentionally empty at initialization.

When we start populating it, expect modules like:

- `sft.py` — supervised fine-tuning loop
- `data.py` — training-time data collators
- `trainer.py` — thin wrapper around `transformers.Trainer` or TRL

Training configs live next to task / model configs under `configs/`.
