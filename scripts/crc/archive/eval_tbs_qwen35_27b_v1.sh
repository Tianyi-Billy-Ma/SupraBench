#!/bin/bash
# Evaluate Qwen3.5-27B + EU-PMC LoRA on TBS (compare H/G interaction strength, base prompt).
EVAL_TASK_CONFIG=tbs_base.yaml \
EVAL_MODEL_CONFIG=qwen35_27b_eupmc_lora.yaml \
exec /path/to/SupraBench/scripts/crc/eval_task.sh
