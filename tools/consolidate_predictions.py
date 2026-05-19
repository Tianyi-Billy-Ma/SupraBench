#!/usr/bin/env python3
"""Consolidate Ziming's per-(task, model, method) inference dumps into the
canonical results/task<idx>/<method>/<model>.jsonl layout.

Source layout (extracted from task3_result.zip / task7_result_{1,2}.zip):
    <stage>/<model_dir>/<method>/{run_meta.json, predictions.csv, full_log.jsonl}

Output layout:
    results/task<idx>/<method>/<model>.jsonl

Row schema (the `prompt` field is intentionally dropped):
    task3:
        {task, method, model, id, subtype,
         reference, prediction, response,
         parse_status, error}
    task7:
        {task, method, model, id,
         reference={"letter", "label"},
         prediction={"letter", "label"},
         response, error}
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

METHODS = ("base", "fewshot", "cot")


def iter_model_dirs(stage_dir: Path) -> Iterable[Path]:
    """Yield top-level model directories inside an extracted zip stage.

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


TRANSFORMERS = {
    "task3": transform_task3_row,
    "task7": transform_task7_row,
}


def consolidate_task(
    *,
    task_id: str,
    stage_dirs: list[Path],
    out_root: Path,
) -> dict[str, int]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task3-stage", type=Path, default=Path("/tmp/suprabench_zips/extract_t3"))
    parser.add_argument("--task7-stage-1", type=Path, default=Path("/tmp/suprabench_zips/extract_t7_1/task7_result_1"))
    parser.add_argument("--task7-stage-2", type=Path, default=Path("/tmp/suprabench_zips/extract_t7_2/task7_result_2"))
    parser.add_argument("--out-root", type=Path, default=Path("results"))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean:
        for sub in ("task3", "task7"):
            target = args.out_root / sub
            if target.exists():
                shutil.rmtree(target)

    written_t3 = consolidate_task(task_id="task3", stage_dirs=[args.task3_stage], out_root=args.out_root)
    written_t7 = consolidate_task(
        task_id="task7",
        stage_dirs=[args.task7_stage_1, args.task7_stage_2],
        out_root=args.out_root,
    )

    print(f"task3: wrote {len(written_t3)} files, {sum(written_t3.values())} total rows")
    print(f"task7: wrote {len(written_t7)} files, {sum(written_t7.values())} total rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
