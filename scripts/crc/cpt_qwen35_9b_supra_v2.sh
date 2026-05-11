#!/bin/bash
#$ -q gpu@@yye7_lab
#$ -l gpu_card=4
#$ -N suprabench_cpt_qwen9b_v2
#$ -pe smp 32
#$ -l h_rt=720:00:00
#$ -o logs/$JOB_NAME_$JOB_ID.log
#$ -e logs/$JOB_NAME_$JOB_ID.err
#$ -m abe
#$ -M tma2@nd.edu
# ---------------------------------------------------------------------------
# v2 LoRA CPT on Qwen3.5-9B using the EvoLM-style mix dataset.
# Projected wall: ~5 h on 4x A40 (vs ~15 h for the 27B sibling).
# ---------------------------------------------------------------------------
set -eo pipefail

mkdir -p logs

source /groups/yye7/BILLY/SupraBench/scripts/crc/base.sh

export WANDB_RUN_GROUP=cpt-supra-v2
export WANDB_NAME="cpt-qwen9b-supra-v2-${JOB_ID:-local}"

# peft's FSDP auto-wrap can't infer the transformer block class on plain
# causal-LM bases — hand it the name explicitly. Class confirmed by direct
# model inspection on 2026-05-11.
export FSDP_TRANSFORMER_CLS_TO_WRAP=Qwen3_5DecoderLayer

nvidia-smi || true

"${ACCELERATE}" launch \
  --config_file scripts/crc/accelerate_fsdp.yaml \
  src/train/cpt_lora.py \
  --config configs/train/cpt_qwen35_9b_supra_v2.yaml
