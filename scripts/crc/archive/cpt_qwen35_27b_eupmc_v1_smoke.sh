#!/bin/bash
#$ -q gpu@@anonymous_lab
#$ -l gpu_card=4
#$ -N suprabench_cpt_smoke
#$ -pe smp 32
#$ -l h_rt=01:30:00
#$ -o logs/$JOB_NAME_$JOB_ID.log
#$ -e logs/$JOB_NAME_$JOB_ID.err
#$ -m abe
#$ -M anonymous@example.org
# ---------------------------------------------------------------------------
# Smoke test for the LoRA CPT pipeline: 20 steps over 256 EU-PMC rows at
# seq_len=1024. Verifies model loads with FSDP, vision tower is frozen,
# LoRA params are trainable, loss decreases. ~15-30 min wall.
# ---------------------------------------------------------------------------
set -eo pipefail   # not -u: ~/.bashrc -> /etc/bashrc references BASHRCSOURCED before defining it

mkdir -p logs

source /path/to/SupraBench/scripts/crc/base.sh

export WANDB_RUN_GROUP=cpt-smoke
export WANDB_NAME="cpt-smoke-${JOB_ID:-local}"

nvidia-smi || true

"${ACCELERATE}" launch \
  --config_file scripts/crc/accelerate_fsdp.yaml \
  src/train/cpt_lora.py \
  --config configs/train/cpt_qwen35_eupmc.yaml \
  --override \
      training.max_steps=20 \
      training.save_steps=50 \
      training.gradient_accumulation_steps=1 \
      dataset.train_rows=256 \
      dataset.seq_len=1024 \
      training.output_dir=outputs/cpt_qwen35_eupmc_smoke
