#!/bin/bash
# Evaluate Qwen3.5-9B + v2 LoRA on Task 7 (solvent compatibility, base prompt).
EVAL_TASK_CONFIG=task7_base.yaml \
EVAL_MODEL_CONFIG=qwen35_9b_supra_v2_lora.yaml \
exec /groups/yye7/BILLY/SupraBench/scripts/crc/eval_task.sh
