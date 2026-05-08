"""SupraBench entry point.

Usage (from the repository root):

    uv run python src/main.py \
        --task-config configs/tasks/task1_base.yaml \
        --model-config configs/models/openrouter_claude_sonnet46.yaml \
        --output-dir outputs/

The main loop:

    1. Load the task config (dataset path, prompt settings, eval metric).
    2. Load the model config (backend, model id, generation kwargs).
    3. Build the dataset, the inference backend, and the evaluator.
    4. Run inference across the dataset, save raw predictions.
    5. Run evaluation, save metrics + per-example results.

Artifacts land in a flat directory named ``<task>_<model>/`` under
``--output-dir`` (default: ``outputs/``).
"""

from __future__ import annotations

import argparse
import json
import sys
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
        "--skip-eval",
        action="store_true",
        help="Run inference only; do not compute metrics.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip inference; re-score an existing predictions.jsonl. Useful "
             "when the evaluator changed but the model outputs are stable.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    task_cfg  = _load_yaml(args.task_config)
    model_cfg = _load_yaml(args.model_config)

    task_name  = task_cfg.get("name")  or args.task_config.stem
    model_name = model_cfg.get("name") or args.model_config.stem

    run_dir = args.output_dir / f"{task_name}_{model_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"

    if not args.eval_only:
        dataset = build_dataset(task_cfg, limit=args.limit)
        backend = build_inference_backend(model_cfg)

        # ── Inference ─────────────────────────────────────────────────────────
        # Materialize the dataset once so batched backends (vLLM continuous
        # batching, etc.) can run all prompts in a single call. Sequential
        # backends still work because the default ``generate_many`` is a loop
        # over ``generate``.
        t0       = time.time()
        examples = list(dataset)
        total    = len(examples)
        print(f"Running {total} examples through {model_cfg.get('backend')} ...")
        predictions = backend.generate_many([ex.prompt for ex in examples])
        if len(predictions) != total:
            raise RuntimeError(
                f"backend returned {len(predictions)} predictions for {total} prompts"
            )

        records = [
            {
                "id":         ex.id,
                "prompt":     ex.prompt,
                "prediction": pred,
                "reference":  ex.reference,
                "metadata":   ex.metadata,
            }
            for ex, pred in zip(examples, predictions)
        ]

        with predictions_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\nInference done in {time.time() - t0:.1f}s  |  output: {predictions_path}")

        if args.skip_eval:
            return
    else:
        if not predictions_path.is_file():
            raise SystemExit(
                f"--eval-only set, but {predictions_path} does not exist. Run "
                "inference first (or drop --eval-only)."
            )
        print(f"--eval-only: re-scoring {predictions_path}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    evaluator = build_evaluator(task_cfg)
    metrics   = evaluator.evaluate(predictions_path)

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    print(f"Metrics saved → {metrics_path}")

    scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    for k, v in scalar.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
