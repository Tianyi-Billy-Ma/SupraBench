#!/bin/bash
#$ -q gpu@@yye7_lab
#$ -l gpu_card=4
#$ -N suprabench_cpt
#$ -pe smp 32
#$ -l h_rt=720:00:00
#$ -o logs/$JOB_NAME_$JOB_ID.log
#$ -e logs/$JOB_NAME_$JOB_ID.err
#$ -m abe
#$ -M tma2@nd.edu
# ---------------------------------------------------------------------------
# Full LoRA continued pretraining of Qwen3.5-27B on EU-PMC (filtered split,
# 133,867 articles). 4× A40, FSDP_FULL_SHARD, vision tower frozen.
#
# Walltime: 720 h (queue maximum on yye7_lab). Projected total wall is ~64 h,
# so the run finishes naturally without a SIGTERM. The SGE directives above
# are also inert when this script is dispatched via `autoexp submit` — bash
# ignores them, AutoExp's outer qsub already owns the GPU allocation.
# ---------------------------------------------------------------------------
set -euo pipefail

mkdir -p logs

source /groups/yye7/BILLY/SupraBench/scripts/crc/base.sh

export WANDB_RUN_GROUP=cpt-eupmc
export WANDB_NAME="cpt-eupmc-${JOB_ID:-local}"

nvidia-smi || true

"${ACCELERATE}" launch \
  --config_file scripts/crc/accelerate_fsdp.yaml \
  src/train/cpt_lora.py \
  --config configs/train/cpt_qwen35_eupmc.yaml
