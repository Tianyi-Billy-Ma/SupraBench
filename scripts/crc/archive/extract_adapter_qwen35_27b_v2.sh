#!/bin/bash
# Convert the v2 FSDP-sharded checkpoint into a PEFT adapter. Submit via:
#   autoexp submit --gpus 1 --name extract_adapter_v2 --cwd /path/to/SupraBench scripts/crc/extract_adapter_v2.sh
# 1 GPU is reserved out of caution but the tool runs entirely on CPU
# (~50–60 GB RAM to consolidate the DCP shards).
set -eo pipefail

mkdir -p logs

source /path/to/SupraBench/scripts/crc/base.sh

CHECKPOINT="/path/to/SupraBench/outputs/cpt_qwen35_supra_v2/checkpoint-302"
OUTPUT="/path/to/SupraBench/outputs/cpt_qwen35_supra_v2/adapter"

"${PYTHON}" tools/fsdp_to_peft_adapter.py \
  --checkpoint "${CHECKPOINT}" \
  --output     "${OUTPUT}" \
  --base-model Qwen/Qwen3.5-27B \
  --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --base-path model.language_model \
  --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
