#!/bin/bash
# eval.sh — run bap/tbs benchmarks via the OpenRouter API.
#
# Usage:
#   export OPENROUTER_API_KEY=sk-or-v1-...
#   bash scripts/eval.sh bap
#   bash scripts/eval.sh tbs
#
# nohup bash scripts/eval.sh bap > outputs/eval_bap.log 2>&1 &
#
# IMPORTANT: set OPENROUTER_API_KEY in your environment — do NOT hardcode it here.

set -euo pipefail

TASK=${1:-"bap"}
PYTHON=${PYTHON:-python}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_CONFIGS=(
    "configs/models/openrouter_gpt54mini.yaml"
    "configs/models/openrouter_gpt54nano_xhigh.yaml"
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

echo "=== SupraBench (${TASK}) ==="
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
            --output-dir   outputs/
        echo
    done
done

echo "=== All runs done: $(date) ==="
