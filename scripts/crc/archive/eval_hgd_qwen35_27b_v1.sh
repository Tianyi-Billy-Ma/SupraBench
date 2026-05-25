#!/bin/bash
# Evaluate Qwen3.5-27B + EU-PMC LoRA on HGD (explain host/guest properties, base prompt).
EVAL_TASK_CONFIG=hgd_base.yaml \
EVAL_MODEL_CONFIG=qwen35_27b_eupmc_lora.yaml \
exec /path/to/SupraBench/scripts/crc/eval_task.sh
