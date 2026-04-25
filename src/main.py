"""SupraBench entry point.

Usage (from the repository root):

    uv run python src/main.py \
        --task-config configs/tasks/task1.yaml \
        --model-config configs/models/qwen3.yaml \
        --output-dir outputs/

The main loop:

    1. Load the task config (dataset path, prompt settings, eval metric).
    2. Load the model config (backend, model id, generation kwargs).
    3. Build the dataset, the inference backend, and the evaluator.
    4. Run inference across the dataset, save raw predictions.
    5. Run evaluation, save metrics + per-example results.

Artifacts land in a flat directory named ``<task>_<model>/`` under
``--output-dir`` (default: ``outputs/``).

Each step is dispatched through a small registry so new tasks / models /
evaluators can be added without touching this file.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    task_cfg = _load_yaml(args.task_config)
    model_cfg = _load_yaml(args.model_config)

    task_name = task_cfg.get("name") or args.task_config.stem
    model_name = model_cfg.get("name") or args.model_config.stem

    run_dir = args.output_dir / f"{task_name}_{model_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(task_cfg, limit=args.limit)
    backend = build_inference_backend(model_cfg)

    predictions_path = run_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as fh:
        for example in dataset:
            prediction = backend.generate(example.prompt)
            record = {
                "id": example.id,
                "prompt": example.prompt,
                "prediction": prediction,
                "reference": example.reference,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.skip_eval:
        return

    evaluator = build_evaluator(task_cfg)
    metrics = evaluator.evaluate(predictions_path)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
