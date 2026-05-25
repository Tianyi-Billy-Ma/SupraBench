#!/bin/bash
# Evaluate Qwen3.5-27B + v2 LoRA on BAP with guided decoding.
# Tests the hypothesis that v2's format collapse is the only thing breaking
# eval — if guided decoding alone recovers MAE < 1.725 baseline, the LoRA
# already knew the chemistry and we ship v2+guided as our method.
EVAL_TASK_CONFIG=bap_base.yaml \
EVAL_MODEL_CONFIG=qwen35_27b_supra_v2_lora_guided.yaml \
exec /path/to/SupraBench/scripts/crc/eval_task.sh
