#!/bin/bash
# Evaluate Qwen3.5-27B + EU-PMC LoRA on Task 7 (solvent compatibility classification, base prompt).
EVAL_TASK_CONFIG=task7_base.yaml \
EVAL_MODEL_CONFIG=qwen35_27b_eupmc_lora_vllm.yaml \
exec /groups/yye7/BILLY/SupraBench/scripts/crc/eval_task.sh
