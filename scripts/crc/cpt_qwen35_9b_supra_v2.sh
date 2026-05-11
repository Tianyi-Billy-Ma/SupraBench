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

nvidia-smi || true

# accelerate launch translates the YAML's fsdp_transformer_layer_cls_to_wrap
# into the FSDP_TRANSFORMER_CLS_TO_WRAP env var inside the workers, so a
# shell-level export is overridden. The shared YAML lists
# "Qwen3_5DecoderLayer,Qwen3_5VisionBlock" (correct for the 27B VLM); the 9B
# is a plain causal LM and has no VisionBlock instances, so peft's auto-wrap
# raises. Pass the override via CLI flag, which takes precedence.
"${ACCELERATE}" launch \
  --config_file scripts/crc/accelerate_fsdp.yaml \
  --fsdp_transformer_layer_cls_to_wrap Qwen3_5DecoderLayer \
  src/train/cpt_lora.py \
  --config configs/train/cpt_qwen35_9b_supra_v2.yaml
