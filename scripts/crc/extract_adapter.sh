#!/bin/bash
# Parameterized DCP -> PEFT adapter extractor.
#
# Reads these env vars (all required unless noted):
#     CHECKPOINT     HF Trainer FSDP checkpoint dir (contains pytorch_model_fsdp_0/)
#     OUTPUT         Where to write adapter_config.json + adapter_model.safetensors
#     BASE_MODEL     HF id of the base model (recorded into adapter_config)
#     LORA_R         LoRA rank used during training (recipe-dependent, no default)
#     LORA_ALPHA     LoRA alpha used during training (recipe-dependent, no default)
#     BASE_PATH      Module-path prefix for LoRA targets. Empty string for plain
#                    causal-LM bases (Qwen3.5-9B, Llama-3.1-8B); use
#                    "model.language_model" for the Qwen3.5-27B VLM.
#
# Optional:
#     LORA_DROPOUT    (default 0.05)
#     TARGET_MODULES  (default "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj")
#
# Don't call this directly via autoexp; use the submit_extract.sh helper, which
# wires the env vars from a recipe name and produces a per-submission launcher.
#
# Runs single-process (no distributed init); needs ~50-60 GB CPU RAM to hold
# the consolidated state dict in bf16 before filtering. 1 GPU is reserved out
# of caution but the tool runs entirely on CPU.
set -eo pipefail

mkdir -p logs

source /groups/yye7/BILLY/SupraBench/scripts/crc/base.sh

: "${CHECKPOINT:?CHECKPOINT must be set (e.g. outputs/cpt_qwen35_supra_v2/checkpoint-302)}"
: "${OUTPUT:?OUTPUT must be set (target adapter dir)}"
: "${BASE_MODEL:?BASE_MODEL must be set (e.g. Qwen/Qwen3.5-27B)}"
: "${LORA_R:?LORA_R must be set (LoRA rank used during training)}"
: "${LORA_ALPHA:?LORA_ALPHA must be set (LoRA alpha used during training)}"
: "${BASE_PATH=}"   # may be empty for plain causal-LM bases
: "${LORA_DROPOUT:=0.05}"
: "${TARGET_MODULES:=q_proj k_proj v_proj o_proj gate_proj up_proj down_proj}"

echo "[extract_adapter]"
echo "  CHECKPOINT     = ${CHECKPOINT}"
echo "  OUTPUT         = ${OUTPUT}"
echo "  BASE_MODEL     = ${BASE_MODEL}"
echo "  LORA_R         = ${LORA_R}"
echo "  LORA_ALPHA     = ${LORA_ALPHA}"
echo "  BASE_PATH      = '${BASE_PATH}'"
echo "  LORA_DROPOUT   = ${LORA_DROPOUT}"
echo "  TARGET_MODULES = ${TARGET_MODULES}"

"${PYTHON}" tools/fsdp_to_peft_adapter.py \
  --checkpoint "${CHECKPOINT}" \
  --output     "${OUTPUT}" \
  --base-model "${BASE_MODEL}" \
  --lora-r "${LORA_R}" --lora-alpha "${LORA_ALPHA}" --lora-dropout "${LORA_DROPOUT}" \
  --base-path "${BASE_PATH}" \
  --target-modules ${TARGET_MODULES}
