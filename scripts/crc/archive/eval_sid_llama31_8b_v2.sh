#!/bin/bash
# Evaluate Llama-3.1-8B + v2 LoRA on SID (solvent compatibility, base prompt).
EVAL_TASK_CONFIG=sid_base.yaml \
EVAL_MODEL_CONFIG=llama31_8b_supra_v2_lora.yaml \
exec /path/to/SupraBench/scripts/crc/eval_task.sh
