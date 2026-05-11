#!/bin/bash
#$ -q gpu@@yye7_lab
#$ -l gpu_card=4
#$ -N suprabench_cpt_llama8b_v2
#$ -pe smp 32
#$ -l h_rt=720:00:00
#$ -o logs/$JOB_NAME_$JOB_ID.log
#$ -e logs/$JOB_NAME_$JOB_ID.err
#$ -m abe
#$ -M tma2@nd.edu
# ---------------------------------------------------------------------------
# v2 LoRA CPT on Llama-3.1-8B using the EvoLM-style mix dataset.
# Projected wall: ~5 h on 4x A40. Tests whether v2's regression is
# Qwen-contamination-specific or recipe-specific.
# ---------------------------------------------------------------------------
set -eo pipefail

mkdir -p logs

source /groups/yye7/BILLY/SupraBench/scripts/crc/base.sh

export WANDB_RUN_GROUP=cpt-supra-v2
export WANDB_NAME="cpt-llama8b-supra-v2-${JOB_ID:-local}"

nvidia-smi || true

# accelerate launch translates the YAML's fsdp_transformer_layer_cls_to_wrap
# into FSDP_TRANSFORMER_CLS_TO_WRAP inside workers, so shell-level exports
# get overridden. CLI flag takes precedence over both YAML and env var.
"${ACCELERATE}" launch \
  --config_file scripts/crc/accelerate_fsdp.yaml \
  --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
  src/train/cpt_lora.py \
  --config configs/train/cpt_llama31_8b_supra_v2.yaml
