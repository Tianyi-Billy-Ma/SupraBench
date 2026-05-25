#!/bin/bash
# Evaluate Qwen3.5-9B + v2 LoRA on TBS (host classification, base prompt).
EVAL_TASK_CONFIG=tbs_base.yaml \
EVAL_MODEL_CONFIG=qwen35_9b_supra_v2_lora.yaml \
exec /path/to/SupraBench/scripts/crc/eval_task.sh
