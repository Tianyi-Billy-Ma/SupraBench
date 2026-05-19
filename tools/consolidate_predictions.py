#!/usr/bin/env python3
"""Consolidate Ziming's per-(task, model, method) inference dumps into the
canonical results/task<idx>/<method>/<model>.jsonl layout.

Two source layouts are supported:

(A) Task 3 / Task 7 dumps (task3_result.zip, task7_result_{1,2}.zip):
        <stage>/<model_dir>/<method>/{run_meta.json, predictions.csv, full_log.jsonl}

(B) Task 1 / Task 2 dumps (task1_task2_results.zip):
        <stage>/task<X>_v<N>/results_task<X>_v<N>_<model>.jsonl
    where v1 = base, v2 = fewshot, v3 = cot. No run_meta sidecar.

Output layout (uniform):
    results/task<idx>/<method>/<model>.jsonl

Row schemas (prompt fields dropped; per-row dataset metadata kept so
files are self-describing for slice analysis):
    task1: {task, method, model, id, host_name, guest_name,
            reference (float), prediction (float|null), response}
    task2: {task, method, model, id, host_name,
            options [4], options_logka [4],
            reference={letter, molecule, logka},
            prediction={letter},
            response}
    task3: {task, method, model, id, subtype,
            reference, prediction, response,
            parse_status, error}
    task7: {task, method, model, id,
            reference={letter, label},
            prediction={letter, label},
            response, error}
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

METHODS = ("base", "fewshot", "cot")
T12_METHOD_BY_VERSION = {"v1": "base", "v2": "fewshot", "v3": "cot"}

# Map T1/T2 filename slugs to the canonical names used by T3/T7 outputs.
T12_MODEL_CANONICAL = {
    "qwen3_5_9b": "qwen3.5-9b",
    "qwen3_5_27b": "qwen3.5-27b",
    "llama_3_1_8b": "llama-3.1-8b-instruct",
    "llama_3_1_70b": "llama-3.1-70b-instruct",
    "gpt_5_4_mini": "gpt-5.4-mini_nothinking",
    "gpt_5_4_nano_xhigh": "gpt-5.4-nano_xhigh",
    "gemini_3_flash_no_thinking": "gemini-3-flash-preview_nothinking",
    "deepseek_chat": "deepseek-v4-pro",
    "claude_sonnet_4_6": "claude-sonnet-4.6",
}


def iter_model_dirs(stage_dir: Path) -> Iterable[Path]:
    """Yield top-level model directories inside an extracted T3/T7 zip stage.

    Handles both flat layouts and zips that wrap everything in a single
    same-named folder.
    """
    if not stage_dir.is_dir():
        return
    children = [p for p in stage_dir.iterdir() if p.is_dir() and not p.name.startswith("__")]
    if len(children) == 1 and children[0].name == stage_dir.name:
        yield from iter_model_dirs(children[0])
        return
    for p in children:
        if p.name.startswith("__"):
            continue
        yield p


def transform_task3_row(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": rec.get("id"),
        "subtype": rec.get("subtype"),
        "reference": rec.get("gold_answer"),
        "prediction": rec.get("pred_answer"),
        "response": rec.get("response"),
        "parse_status": rec.get("parse_status"),
        "error": rec.get("error"),
    }


def transform_task7_row(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": rec.get("interaction_id"),
        "reference": {
            "letter": rec.get("true_letter"),
            "label": rec.get("true_label"),
        },
        "prediction": {
            "letter": rec.get("pred_letter"),
            "label": rec.get("pred_label"),
        },
        "response": rec.get("response"),
        "error": rec.get("error"),
    }


def transform_task1_row(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": rec.get("id"),
        "host_name": rec.get("host_name"),
        "guest_name": rec.get("molecule"),
        "reference": rec.get("answer"),
        "prediction": rec.get("pred"),
        "response": rec.get("raw_output"),
    }


def transform_task2_row(rec: dict[str, Any]) -> dict[str, Any]:
    options = rec.get("options") or []
    logkas = rec.get("options_logka") or []
    gold_letter = rec.get("answer")
    gold_idx = ord(gold_letter) - ord("A") if isinstance(gold_letter, str) and len(gold_letter) == 1 else None
    gold_logka = logkas[gold_idx] if gold_idx is not None and 0 <= gold_idx < len(logkas) else None
    return {
        "id": rec.get("id"),
        "host_name": rec.get("host_name"),
        "options": options,
        "options_logka": logkas,
        "reference": {
            "letter": gold_letter,
            "molecule": rec.get("correct_molecule"),
            "logka": gold_logka,
        },
        "prediction": {"letter": rec.get("pred_letter")},
        "response": rec.get("raw_output"),
    }


TRANSFORMERS = {
    "task1": transform_task1_row,
    "task2": transform_task2_row,
    "task3": transform_task3_row,
    "task7": transform_task7_row,
}


def consolidate_t37_task(
    *,
    task_id: str,
    stage_dirs: list[Path],
    out_root: Path,
) -> dict[str, int]:
    """T3/T7 layout: <stage>/<model>/<method>/full_log.jsonl + run_meta.json."""
    transform = TRANSFORMERS[task_id]
    written: dict[str, int] = {}
    seen_keys: set[tuple[str, str]] = set()
    for stage in stage_dirs:
        for model_dir in iter_model_dirs(stage):
            for method in METHODS:
                method_dir = model_dir / method
                full_log = method_dir / "full_log.jsonl"
                run_meta = method_dir / "run_meta.json"
                if not full_log.is_file():
                    continue
                meta = json.loads(run_meta.read_text()) if run_meta.is_file() else {}
                out_name = meta.get("out_name") or model_dir.name
                key = (method, out_name)
                if key in seen_keys:
                    raise RuntimeError(
                        f"Duplicate ({method}, {out_name}) for {task_id}: second source "
                        f"is {method_dir}; first already consumed."
                    )
                seen_keys.add(key)

                target_dir = out_root / task_id / method
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / f"{out_name}.jsonl"
                meta_target = target_dir / f"{out_name}.meta.json"

                if meta:
                    meta_target.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

                count = 0
                with full_log.open() as fin, target.open("w") as fout:
                    for line in fin:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        body = transform(rec)
                        out_rec = {
                            "task": task_id,
                            "method": method,
                            "model": out_name,
                            **body,
                        }
                        fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                        count += 1
                written[f"{task_id}/{method}/{out_name}"] = count
    return written


def consolidate_t12_task(
    *,
    task_id: str,
    stage_dir: Path,
    out_root: Path,
) -> dict[str, int]:
    """T1/T2 layout: <stage>/task<X>_v<N>/results_task<X>_v<N>_<model>.jsonl."""
    transform = TRANSFORMERS[task_id]
    task_num = task_id.replace("task", "")
    written: dict[str, int] = {}
    seen_keys: set[tuple[str, str]] = set()

    for version, method in T12_METHOD_BY_VERSION.items():
        version_dir = stage_dir / f"task{task_num}_{version}"
        if not version_dir.is_dir():
            continue
        prefix = f"results_task{task_num}_{version}_"
        for jsonl_path in sorted(version_dir.glob(f"{prefix}*.jsonl")):
            slug = jsonl_path.stem[len(prefix):]
            out_name = T12_MODEL_CANONICAL.get(slug, slug)
            key = (method, out_name)
            if key in seen_keys:
                raise RuntimeError(
                    f"Duplicate ({method}, {out_name}) for {task_id}: second source "
                    f"is {jsonl_path}."
                )
            seen_keys.add(key)

            target_dir = out_root / task_id / method
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{out_name}.jsonl"

            count = 0
            with jsonl_path.open() as fin, target.open("w") as fout:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    body = transform(rec)
                    out_rec = {
                        "task": task_id,
                        "method": method,
                        "model": out_name,
                        **body,
                    }
                    fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    count += 1
            written[f"{task_id}/{method}/{out_name}"] = count
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task3-stage", type=Path, default=Path("/tmp/suprabench_zips/extract_t3"))
    parser.add_argument("--task7-stage-1", type=Path, default=Path("/tmp/suprabench_zips/extract_t7_1/task7_result_1"))
    parser.add_argument("--task7-stage-2", type=Path, default=Path("/tmp/suprabench_zips/extract_t7_2/task7_result_2"))
    parser.add_argument("--t12-stage", type=Path, default=Path("/tmp/suprabench_zips/extract_t12/task1_task2_results"))
    parser.add_argument("--out-root", type=Path, default=Path("results"))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--only", choices=("t12", "t37"), default=None,
                        help="Consolidate only one source set (default: both).")
    args = parser.parse_args()

    if args.clean:
        for sub in ("task1", "task2", "task3", "task7"):
            target = args.out_root / sub
            if target.exists():
                shutil.rmtree(target)

    totals = {}
    if args.only != "t12":
        totals["task3"] = consolidate_t37_task(task_id="task3", stage_dirs=[args.task3_stage], out_root=args.out_root)
        totals["task7"] = consolidate_t37_task(
            task_id="task7",
            stage_dirs=[args.task7_stage_1, args.task7_stage_2],
            out_root=args.out_root,
        )
    if args.only != "t37":
        totals["task1"] = consolidate_t12_task(task_id="task1", stage_dir=args.t12_stage, out_root=args.out_root)
        totals["task2"] = consolidate_t12_task(task_id="task2", stage_dir=args.t12_stage, out_root=args.out_root)

    for task_id, written in totals.items():
        print(f"{task_id}: wrote {len(written)} files, {sum(written.values())} total rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
