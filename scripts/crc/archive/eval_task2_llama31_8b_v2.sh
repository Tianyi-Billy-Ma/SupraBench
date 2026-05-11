#!/bin/bash
# Evaluate Llama-3.1-8B + v2 LoRA on Task 2 (host classification, base prompt).
EVAL_TASK_CONFIG=task2_base.yaml \
EVAL_MODEL_CONFIG=llama31_8b_supra_v2_lora.yaml \
exec /groups/yye7/BILLY/SupraBench/scripts/crc/eval_task.sh
