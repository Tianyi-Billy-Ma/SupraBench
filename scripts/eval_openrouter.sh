#!/bin/bash
# eval_openrouter.sh — run task1/task2 benchmarks via the OpenRouter API.
#
# Usage:
#   export OPENROUTER_API_KEY=sk-or-v1-...
#   bash scripts/eval_openrouter.sh task1
#   bash scripts/eval_openrouter.sh task2
#
# nohup bash scripts/eval_openrouter.sh task1 > outputs/eval_openrouter_task1.log 2>&1 &
#
# IMPORTANT: set OPENROUTER_API_KEY in your environment — do NOT hardcode it here.

set -euo pipefail

TASK=${1:-"task1"}
PYTHON=${PYTHON:-python}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONCURRENCY=${CONCURRENCY:-8}

MODEL_CONFIGS=(
    "configs/models/openrouter_qwen35_9b.yaml"
    "configs/models/openrouter_qwen35_27b.yaml"
    "configs/models/openrouter_llama31_8b.yaml"
    "configs/models/openrouter_llama31_70b.yaml"
    "configs/models/openrouter_gemini3_flash.yaml"
    "configs/models/openrouter_claude_sonnet46.yaml"
    "configs/models/openrouter_deepseek_v4.yaml"
)

TASK_CONFIGS=(
    "configs/tasks/${TASK}_base.yaml"
    "configs/tasks/${TASK}_fewshot.yaml"
    "configs/tasks/${TASK}_cot.yaml"
)

cd "$REPO_ROOT"
mkdir -p outputs

N_TOTAL=$(( ${#MODEL_CONFIGS[@]} * ${#TASK_CONFIGS[@]} ))
RUN=0

echo "=== SupraBench — OpenRouter (${TASK}) ==="
echo "Models   : ${#MODEL_CONFIGS[@]}"
echo "Variants : ${#TASK_CONFIGS[@]}"
echo "Total    : ${N_TOTAL} runs"
echo "Time     : $(date)"
echo

for MODEL_CFG in "${MODEL_CONFIGS[@]}"; do
    for TASK_CFG in "${TASK_CONFIGS[@]}"; do
        RUN=$(( RUN + 1 ))
        echo "── [${RUN}/${N_TOTAL}] ${MODEL_CFG} × ${TASK_CFG} ──"
        $PYTHON src/main.py \
            --task-config  "${TASK_CFG}" \
            --model-config "${MODEL_CFG}" \
            --output-dir   outputs/ \
            --concurrency  "${CONCURRENCY}"
        echo
    done
done

echo "=== All runs done: $(date) ==="
