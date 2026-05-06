"""SupraBench entry point.

Usage (from the repository root):

    uv run python src/main.py \
        --task-config configs/tasks/task1_base.yaml \
        --model-config configs/models/openai_gpt54mini.yaml \
        --output-dir outputs/

    # Concurrent API calls (recommended for OpenAI / OpenRouter backends)
    uv run python src/main.py \
        --task-config configs/tasks/task1_base.yaml \
        --model-config configs/models/openai_gpt54mini.yaml \
        --concurrency 8

The main loop:

    1. Load the task config (dataset path, prompt settings, eval metric).
    2. Load the model config (backend, model id, generation kwargs).
    3. Build the dataset, the inference backend, and the evaluator.
    4. Run inference across the dataset, save raw predictions.
       - If predictions.jsonl already exists, resume from where it left off.
       - When --concurrency > 1, requests are issued in parallel via
         ThreadPoolExecutor (useful for OpenAI / OpenRouter backends).
    5. Run evaluation, save metrics + per-example results.

Artifacts land in a flat directory named ``<task>_<model>/`` under
``--output-dir`` (default: ``outputs/``).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
import time
from pathlib import Path

import yaml

# Ensure `datasets`, `eval`, `inference`, `models`, `extras`, `templates`
# are importable as top-level packages when running `python src/main.py`.
SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datasets import build_dataset  # noqa: E402
from eval import build_evaluator  # noqa: E402
from extras.constants import DEFAULT_OUTPUT_DIR  # noqa: E402
from inference import build_inference_backend  # noqa: E402


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a SupraBench task against a model.")
    parser.add_argument(
        "--task-config",
        type=Path,
        required=True,
        help="Path to a task YAML under configs/tasks/.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to a model YAML under configs/models/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory where predictions and metrics are written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of examples to run (useful for smoke tests).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of parallel API requests. Use >1 for OpenAI/OpenRouter backends.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Run inference only; do not compute metrics.",
    )
    return parser.parse_args(argv)


def _load_existing(predictions_path: Path) -> dict[str, dict]:
    """Return {id: record} for all records with a non-empty prediction."""
    if not predictions_path.exists():
        return {}
    existing: dict[str, dict] = {}
    with predictions_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("prediction") is not None and str(rec.get("prediction", "")).strip():
                existing[rec["id"]] = rec
    return existing


def run(args: argparse.Namespace) -> None:
    task_cfg  = _load_yaml(args.task_config)
    model_cfg = _load_yaml(args.model_config)

    task_name  = task_cfg.get("name")  or args.task_config.stem
    model_name = model_cfg.get("name") or args.model_config.stem

    run_dir = args.output_dir / f"{task_name}_{model_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(task_cfg, limit=args.limit)
    backend = build_inference_backend(model_cfg)

    predictions_path = run_dir / "predictions.jsonl"

    # ── Resume: skip examples that already have a prediction ─────────────────
    existing   = _load_existing(predictions_path)
    all_examples = list(dataset)
    todo        = [e for e in all_examples if e.id not in existing]

    if existing:
        print(f"Resuming: {len(existing)} done, {len(todo)} remaining.")

    # ── Inference ─────────────────────────────────────────────────────────────
    t0      = time.time()
    lock    = threading.Lock()
    counter = [0]
    total   = len(todo)

    def _generate_one(example):
        prediction = backend.generate(example.prompt)
        with lock:
            counter[0] += 1
            done = counter[0]
        if done % 32 == 0 or done == total:
            print(f"  {done}/{total}  ({time.time() - t0:.0f}s)")
        return {
            "id":         example.id,
            "prompt":     example.prompt,
            "prediction": prediction,
            "reference":  example.reference,
            "metadata":   example.metadata,
        }

    if args.concurrency > 1 and todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            new_records = list(ex.map(_generate_one, todo))
    else:
        new_records = [_generate_one(e) for e in todo]

    # Merge and write in original order
    id_to_rec = {**existing, **{r["id"]: r for r in new_records}}
    with predictions_path.open("w", encoding="utf-8") as fh:
        for example in all_examples:
            fh.write(json.dumps(id_to_rec[example.id], ensure_ascii=False) + "\n")

    print(f"\nInference done in {time.time() - t0:.1f}s  |  output: {predictions_path}")

    if args.skip_eval:
        return

    # ── Evaluation ────────────────────────────────────────────────────────────
    evaluator = build_evaluator(task_cfg)
    metrics   = evaluator.evaluate(predictions_path)

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    print(f"Metrics saved → {metrics_path}")

    # Pretty-print top-level scalar metrics
    scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    for k, v in scalar.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
