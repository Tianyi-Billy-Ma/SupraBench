#!/bin/bash
# Consolidate the FSDP-sharded Qwen3.5-9B v2 checkpoint into PEFT format.
set -eo pipefail

mkdir -p logs

source /path/to/SupraBench/scripts/crc/base.sh

OUT="/path/to/SupraBench/outputs/cpt_qwen35_9b_supra_v2"
CKPT="$(ls -d ${OUT}/checkpoint-* 2>/dev/null | sort -V | tail -1)"
[[ -n "$CKPT" ]] || { echo "no checkpoint found under $OUT"; exit 1; }

"${PYTHON}" tools/fsdp_to_peft_adapter.py \
  --checkpoint "${CKPT}" \
  --output     "${OUT}/adapter" \
  --base-model Qwen/Qwen3.5-9B \
  --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --base-path "" \
  --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
