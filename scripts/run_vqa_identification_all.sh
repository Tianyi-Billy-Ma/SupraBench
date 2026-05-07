#!/usr/bin/env bash
# Run vqa_identification on every VQA-capable model under each of the
# three prompting modes (base / fewshot / cot). 6 models × 3 modes = 18 runs.
#
# Requirements:
#   * Repo root as cwd (cd /path/to/SupraBench && bash scripts/run_vqa_identification_all.sh)
#   * GPU node with vLLM + Qwen3-VL-4B for the local model (qwen3-vl-4b)
#   * .env with OPENROUTER_API_KEY filled in for the OpenRouter models
#   * Deps installed: uv sync --extra vllm --extra hf --extra api --extra vqa
#
# Outputs: outputs/vqa_identification_<mode>_<model>/{predictions.jsonl,metrics.json}
#
# Run failures are logged but do not abort the loop (one bad model
# shouldn't kill the whole sweep). Check the trailing summary.

set -u
set -o pipefail

# Force unbuffered Python output so tqdm progress + [run] log lines flush
# to the per-run log file in real time (otherwise stdout/stderr buffer
# until the process exits, and `tail -f` shows nothing for hours).
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."   # repo root, regardless of where invoked from

LOG_DIR="logs/vqa_identification_sweep_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.tsv"
printf "model\tmode\tstatus\tlog\n" > "$SUMMARY"

# Models with verified vision support on OpenRouter (2026-04-30) plus the
# local vLLM Qwen3-VL. Llama-3.1 and DeepSeek-v4-pro are text-only and
# would 422 on image_url content — excluded.
MODELS=(
    # qwen3-vl-4b                       # local vLLM
    openrouter-qwen3.5-9b
    openrouter-qwen3.5-27b
    openrouter-gpt-5.4-nano
    openrouter-gemini-3-flash
    openrouter-claude-sonnet-4.6
)

MODES=(base fewshot cot)

# --- ensure deps are installed (idempotent: uv re-syncs only if pyproject changed) ---
# Always need api (openai SDK + python-dotenv) and vqa (pillow + rdkit).
# If qwen3-vl-4b is in MODELS (uncommented), also pull vllm + hf — those are GB-scale
# downloads and pointless when only the OpenRouter models are running.
EXTRAS=(--extra api --extra vqa)
for m in "${MODELS[@]}"; do
    if [[ "$m" == "qwen3-vl-4b" ]]; then
        EXTRAS+=(--extra vllm --extra hf)
        break
    fi
done

echo "[$(date +%H:%M:%S)] uv sync ${EXTRAS[*]}"
uv sync "${EXTRAS[@]}"

# Optional smoke-test cap: `LIMIT=5 bash scripts/run_vqa_identification_all.sh`
# runs only 5 examples per model/mode to verify all 18 paths work end-to-end
# before kicking off the full ~1773-row sweep.
LIMIT_FLAG=()
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_FLAG=(--limit "$LIMIT")
    echo "[note] LIMIT=$LIMIT — running smoke test, not full sweep"
fi

# Concurrency: OpenRouter is HTTP-bound (1773 rows × few seconds = hours
# serial) so 16-way parallel ≈ 16× speedup. Local vLLM is GPU-bound and
# shouldn't be hit from multiple Python threads — keep it at 1 (vLLM
# batches internally).
concurrency_for() {
    case "$1" in
        qwen3-vl-4b) echo 1 ;;
        openrouter-*) echo 16 ;;
        *)            echo 1 ;;
    esac
}

for model in "${MODELS[@]}"; do
    conc=$(concurrency_for "$model")
    for mode in "${MODES[@]}"; do
        tag="${model}__${mode}"
        log="$LOG_DIR/${tag}.log"
        echo "=========================================================="
        echo "[$(date +%H:%M:%S)] running ${model}  mode=${mode}  concurrency=${conc}"
        echo "log → ${log}"
        echo "=========================================================="

        if uv run python src/main.py \
            --task-config "configs/tasks/vqa_identification_${mode}.yaml" \
            --model-config "configs/models/${model}.yaml" \
            --concurrency "$conc" \
            "${LIMIT_FLAG[@]}" \
            > "$log" 2>&1
        then
            status="OK"
        else
            status="FAIL"
            echo "[FAIL] ${model} / ${mode} — see ${log}"
        fi
        printf "%s\t%s\t%s\t%s\n" "$model" "$mode" "$status" "$log" >> "$SUMMARY"
    done
done

echo
echo "=========================================================="
echo "sweep done — summary at $SUMMARY"
echo "=========================================================="
column -t -s $'\t' "$SUMMARY"
