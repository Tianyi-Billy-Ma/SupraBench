#!/bin/bash
#$ -q gpu@@anonymous_lab
#$ -l gpu_card=4
#$ -N suprabench_cpt_v2_smoke
#$ -pe smp 32
#$ -l h_rt=01:30:00
#$ -o logs/$JOB_NAME_$JOB_ID.log
#$ -e logs/$JOB_NAME_$JOB_ID.err
#$ -m abe
#$ -M anonymous@example.org
# ---------------------------------------------------------------------------
# Smoke test for v2: 20 steps over 256 mix rows at seq_len=1024. Verifies
# the v2 mix dataset loads correctly across all three streams and LoRA
# params are trainable. ~15-30 min wall.
# ---------------------------------------------------------------------------
set -eo pipefail

mkdir -p logs

source /path/to/SupraBench/scripts/crc/base.sh

export WANDB_RUN_GROUP=cpt-supra-v2-smoke
export WANDB_NAME="cpt-supra-v2-smoke-${JOB_ID:-local}"

nvidia-smi || true

"${ACCELERATE}" launch \
  --config_file scripts/crc/accelerate_fsdp.yaml \
  src/train/cpt_lora.py \
  --config configs/train/cpt_qwen35_supra_v2.yaml \
  --override \
      training.max_steps=20 \
      training.save_steps=50 \
      training.gradient_accumulation_steps=1 \
      dataset.train_rows=256 \
      dataset.seq_len=1024 \
      training.output_dir=outputs/cpt_qwen35_supra_v2_smoke
