#!/bin/bash
# Evaluate Qwen3.5-27B + EU-PMC LoRA on Task 2 (compare H/G interaction strength, base prompt).
EVAL_TASK_CONFIG=task2_base.yaml \
EVAL_MODEL_CONFIG=qwen35_27b_eupmc_lora.yaml \
exec /groups/yye7/BILLY/SupraBench/scripts/crc/eval_task.sh
