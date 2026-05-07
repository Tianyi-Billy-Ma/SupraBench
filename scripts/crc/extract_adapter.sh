#!/bin/bash
# Convert the FSDP-sharded checkpoint to PEFT adapter format. Submit via:
#   autoexp submit --gpus 1 --name extract_adapter --cwd /groups/yye7/BILLY/SupraBench scripts/crc/extract_adapter.sh
# 1 GPU is reserved out of caution but the tool runs entirely on CPU.
set -eo pipefail   # not -u: ~/.bashrc -> /etc/bashrc references BASHRCSOURCED

mkdir -p logs

source /groups/yye7/BILLY/SupraBench/scripts/crc/base.sh

CHECKPOINT="/groups/yye7/BILLY/SupraBench/outputs/cpt_qwen35_eupmc/checkpoint-1302"
OUTPUT="/groups/yye7/BILLY/SupraBench/outputs/cpt_qwen35_eupmc/adapter"

"${PYTHON}" tools/fsdp_to_peft_adapter.py \
  --checkpoint "${CHECKPOINT}" \
  --output     "${OUTPUT}" \
  --base-model Qwen/Qwen3.5-27B \
  --lora-r 64 --lora-alpha 128 --lora-dropout 0.05 \
  --base-path model.language_model \
  --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
