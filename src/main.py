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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Iterable, Iterator

import yaml
from tqdm import tqdm

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
        "--concurrency",
        type=int,
        default=1,
        help=(
            "How many examples to keep in flight against the backend. "
            "Use 1 for local GPU backends (vLLM); use 8-32 for HTTP API "
            "backends (OpenRouter etc.) where latency dominates."
        ),
    )
    return parser.parse_args(argv)


def _parallel_imap_unordered(
    fn: Callable,
    items: Iterable,
    max_workers: int,
    max_pending: int | None = None,
) -> Iterator:
    """Stream-yield results from up to ``max_workers`` concurrent calls.

    At most ``max_pending`` (default 2 * max_workers) tasks are in flight at
    once, so memory stays bounded for large datasets where each item carries
    images / fewshot demos. Results yield in completion order — predictions
    are keyed by ``Example.id`` downstream so order is irrelevant.
    """
    if max_pending is None:
        max_pending = max(max_workers * 2, max_workers + 4)

    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending: set = set()
        for _ in range(max_pending):
            try:
                pending.add(executor.submit(fn, next(iterator)))
            except StopIteration:
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                yield fut.result()
                try:
                    pending.add(executor.submit(fn, next(iterator)))
                except StopIteration:
                    pass


def run(args: argparse.Namespace) -> None:
    task_cfg = _load_yaml(args.task_config)
    model_cfg = _load_yaml(args.model_config)

    task_name = task_cfg.get("name") or args.task_config.stem
    model_name = model_cfg.get("name") or args.model_config.stem

    run_dir = args.output_dir / f"{task_name}_{model_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] task={task_name} model={model_name} -> {run_dir}", flush=True)

    print(f"[run] building dataset...", flush=True)
    dataset = build_dataset(task_cfg, limit=args.limit)
    print(f"[run] building inference backend...", flush=True)
    backend = build_inference_backend(model_cfg)

    predictions_path = run_dir / "predictions.jsonl"
    concurrency = max(1, int(args.concurrency))
    print(
        f"[run] starting inference -> {predictions_path}  concurrency={concurrency}",
        flush=True,
    )

    def _infer(example):
        return example, backend.generate(example)

    with predictions_path.open("w", encoding="utf-8") as fh:
        # tqdm renders to stderr; with PYTHONUNBUFFERED=1 the bar shows live
        # in `tail -f`.
        bar = tqdm(desc=f"{task_name}/{model_name}", unit="ex")
        if concurrency == 1:
            results = ((ex, backend.generate(ex)) for ex in dataset)
        else:
            results = _parallel_imap_unordered(_infer, dataset, max_workers=concurrency)
        for example, prediction in results:
            record = {
                "id": example.id,
                "prompt": example.prompt,
                "prediction": prediction,
                "reference": example.reference,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()  # so `tail -f predictions.jsonl` shows live progress too
            bar.update(1)
        bar.close()

    if args.skip_eval:
        return

    print(f"[run] inference done; computing metrics...", flush=True)
    evaluator = build_evaluator(task_cfg)
    metrics = evaluator.evaluate(predictions_path)
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    print(f"[run] metrics -> {metrics_path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
