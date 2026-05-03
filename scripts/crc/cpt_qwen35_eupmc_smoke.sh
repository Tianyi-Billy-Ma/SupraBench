#!/bin/bash
#$ -q gpu@@yye7_lab
#$ -l gpu_card=4
#$ -N suprabench_cpt_smoke
#$ -pe smp 32
#$ -l h_rt=01:30:00
#$ -o logs/$JOB_NAME_$JOB_ID.log
#$ -e logs/$JOB_NAME_$JOB_ID.err
#$ -m abe
#$ -M tma2@nd.edu
# ---------------------------------------------------------------------------
# Smoke test for the LoRA CPT pipeline: 20 steps over 256 EU-PMC rows at
# seq_len=1024. Verifies model loads with FSDP, vision tower is frozen,
# LoRA params are trainable, loss decreases. ~15-30 min wall.
# ---------------------------------------------------------------------------
set -euo pipefail

mkdir -p logs

source /groups/yye7/BILLY/SupraBench/scripts/crc/base.sh

export WANDB_RUN_GROUP=cpt-smoke
export WANDB_NAME="cpt-smoke-${JOB_ID:-local}"
export DEEPSPEED_CONFIG_FILE=scripts/crc/ds_config_zero3.json

nvidia-smi || true

"${ACCELERATE}" launch \
  --config_file scripts/crc/accelerate_deepspeed.yaml \
  src/train/cpt_lora.py \
  --config configs/train/cpt_qwen35_eupmc.yaml \
  --override \
      training.max_steps=20 \
      training.save_steps=50 \
      training.gradient_checkpointing=false \
      lora.dropout=0.0 \
      dataset.train_rows=2048 \
      dataset.seq_len=1024 \
      training.output_dir=outputs/cpt_qwen35_eupmc_smoke
