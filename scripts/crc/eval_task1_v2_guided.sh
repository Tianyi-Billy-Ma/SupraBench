#!/bin/bash
# Evaluate Qwen3.5-27B + v2 LoRA on Task 1 with guided decoding.
# Tests the hypothesis that v2's format collapse is the only thing breaking
# eval — if guided decoding alone recovers MAE < 1.725 baseline, the LoRA
# already knew the chemistry and we ship v2+guided as our method.
EVAL_TASK_CONFIG=task1_base.yaml \
EVAL_MODEL_CONFIG=qwen35_27b_supra_v2_lora_guided.yaml \
exec /groups/yye7/BILLY/SupraBench/scripts/crc/eval_task.sh
