#!/bin/bash
#$ -q gpu@@yye7_lab
#$ -l gpu_card=4
#$ -N suprabench_cpt_llama8b_v2_smoke
#$ -pe smp 32
#$ -l h_rt=01:30:00
#$ -o logs/$JOB_NAME_$JOB_ID.log
#$ -e logs/$JOB_NAME_$JOB_ID.err
#$ -m abe
#$ -M tma2@nd.edu
# ---------------------------------------------------------------------------
# Smoke test for the Llama-3.1-8B v2 LoRA CPT: 20 steps over 256 mix rows at
# seq_len=1024. Verifies `arch: causal_lm` works with a non-Qwen tokenizer
# (Llama's BPE, vocab ~128k) and base.
# ---------------------------------------------------------------------------
set -eo pipefail

mkdir -p logs

source /groups/yye7/BILLY/SupraBench/scripts/crc/base.sh

export WANDB_RUN_GROUP=cpt-supra-v2-smoke
export WANDB_NAME="cpt-llama8b-supra-v2-smoke-${JOB_ID:-local}"

# peft's FSDP auto-wrap can't infer the transformer block class on plain
# causal-LM bases — hand it the name explicitly.
export FSDP_TRANSFORMER_CLS_TO_WRAP=LlamaDecoderLayer

nvidia-smi || true

"${ACCELERATE}" launch \
  --config_file scripts/crc/accelerate_fsdp.yaml \
  src/train/cpt_lora.py \
  --config configs/train/cpt_llama31_8b_supra_v2.yaml \
  --override \
      training.max_steps=20 \
      training.save_steps=50 \
      training.gradient_accumulation_steps=1 \
      dataset.train_rows=256 \
      dataset.seq_len=1024 \
      training.output_dir=outputs/cpt_llama31_8b_supra_v2_smoke
