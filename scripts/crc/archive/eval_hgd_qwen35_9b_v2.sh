#!/bin/bash
# Evaluate Qwen3.5-9B + v2 LoRA on HGD (host description, base prompt).
EVAL_TASK_CONFIG=hgd_base.yaml \
EVAL_MODEL_CONFIG=qwen35_9b_supra_v2_lora.yaml \
exec /path/to/SupraBench/scripts/crc/eval_task.sh
