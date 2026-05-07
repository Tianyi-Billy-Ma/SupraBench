#!/bin/bash
# ---------------------------------------------------------------------------
# Generic SupraBench evaluation wrapper for CRC AutoExp dispatch.
#
# Reads two env vars set by the per-task wrapper script:
#   EVAL_TASK_CONFIG   — path under configs/tasks/ (e.g. task1_base.yaml)
#   EVAL_MODEL_CONFIG  — path under configs/models/ (e.g. qwen35_27b_eupmc_lora.yaml)
#
# Run via `autoexp submit --gpus 4 --name <task>_eval scripts/crc/eval_task<n>.sh`.
# AutoExp owns the GPU allocation; this script just sources the SupraBench
# env (HF_HOME, venv, AFS token) and invokes src/main.py.
# ---------------------------------------------------------------------------
set -eo pipefail   # not -u: ~/.bashrc -> /etc/bashrc references BASHRCSOURCED

mkdir -p logs

source /groups/yye7/BILLY/SupraBench/scripts/crc/base.sh

: "${EVAL_TASK_CONFIG:?EVAL_TASK_CONFIG must be set}"
: "${EVAL_MODEL_CONFIG:?EVAL_MODEL_CONFIG must be set}"

echo "[eval_task] task  = ${EVAL_TASK_CONFIG}"
echo "[eval_task] model = ${EVAL_MODEL_CONFIG}"
echo

nvidia-smi || true

"${PYTHON}" src/main.py \
  --task-config  "configs/tasks/${EVAL_TASK_CONFIG}" \
  --model-config "configs/models/${EVAL_MODEL_CONFIG}" \
  --output-dir   "${OUTPUTS_DIR}"
