#!/bin/bash
# Evaluate Qwen3.5-27B + v2 LoRA on Task 1 (Ka prediction, base prompt).
EVAL_TASK_CONFIG=task1_base.yaml \
EVAL_MODEL_CONFIG=qwen35_27b_supra_v2_lora.yaml \
exec /groups/yye7/BILLY/SupraBench/scripts/crc/eval_task.sh
