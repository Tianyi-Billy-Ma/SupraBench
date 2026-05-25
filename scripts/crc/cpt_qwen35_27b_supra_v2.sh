#!/bin/bash
#$ -q gpu@@anonymous_lab
#$ -l gpu_card=4
#$ -N suprabench_cpt_v2
#$ -pe smp 32
#$ -l h_rt=720:00:00
#$ -o logs/$JOB_NAME_$JOB_ID.log
#$ -e logs/$JOB_NAME_$JOB_ID.err
#$ -m abe
#$ -M anonymous@example.org
# ---------------------------------------------------------------------------
# v2 LoRA continued pretraining of Qwen3.5-27B on the EvoLM-style mix:
#   80% mtybilly/EU-PMC[supramolecular] + 15% FineWeb-Edu + 5% Tulu-3 anchor.
# 4× A40, FSDP_FULL_SHARD, vision tower frozen, LoRA r=32 (was 64), lr=1e-5
# (was 2e-5), 1 epoch (was 2). Total mix is ~21M tokens; projected wall ~12-18h.
# ---------------------------------------------------------------------------
set -eo pipefail

mkdir -p logs

source /path/to/SupraBench/scripts/crc/base.sh

export WANDB_RUN_GROUP=cpt-supra-v2
export WANDB_NAME="cpt-supra-v2-${JOB_ID:-local}"

nvidia-smi || true

"${ACCELERATE}" launch \
  --config_file scripts/crc/accelerate_fsdp.yaml \
  src/train/cpt_lora.py \
  --config configs/train/cpt_qwen35_supra_v2.yaml
