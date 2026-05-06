"""
重启未完成的 task3 combos (14 个), 利用 03d_run_inference_openrouter.py 的
interaction_id 断点续跑。每个 combo 一个 detached 进程, log 追加到原文件。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = str(ROOT / "scripts" / "03d_run_inference_openrouter.py")
PY = "/anvil/projects/x-cis260048/phase4_env/bin/python"
PROGRESS_LOG = str(ROOT / "logs" / "experiment_progress.jsonl")
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(ROOT / "scripts"))
from run_openrouter_batch import MODELS  # noqa

MODELS_BY_NAME = {m["out_name"]: m for m in MODELS}

# (out_name, setting)
INCOMPLETE = [
    ("claude-sonnet-4.6", "cot"),
    ("deepseek-v4-pro", "base"),
    ("deepseek-v4-pro", "fewshot"),
    ("deepseek-v4-pro", "cot"),
    ("gemini-3-flash-preview_high", "base"),
    ("gemini-3-flash-preview_high", "fewshot"),
    ("gemini-3-flash-preview_high", "cot"),
    ("gpt-5.5_xhigh", "fewshot"),
    ("gpt-5.5_xhigh", "cot"),
    ("qwen3.5-27b", "base"),
    ("qwen3.5-27b", "fewshot"),
    ("qwen3.5-27b", "cot"),
    ("qwen3.5-9b", "fewshot"),
    ("qwen3.5-9b", "cot"),
]


def build_cmd(mc: dict, setting: str) -> list[str]:
    cmd = [
        PY, "-u", SCRIPT,
        "--task", "task3",
        "--model", mc["model"],
        "--out-name", mc["out_name"],
        "--prompt", setting,
        "--concurrency", str(mc["concurrency"]),
        "--log-path", PROGRESS_LOG,
    ]
    r = mc.get("reasoning")
    if r:
        if r.get("exclude") and r.get("enabled") is False:
            cmd.append("--no-reasoning")
        elif "effort" in r:
            cmd += ["--reasoning-effort", r["effort"]]
        else:
            cmd += ["--reasoning-json", json.dumps(r)]
    return cmd


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY 未设置", file=sys.stderr)
        sys.exit(2)

    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = api_key

    spawned = []
    for out_name, setting in INCOMPLETE:
        mc = MODELS_BY_NAME[out_name]
        log_path = LOG_DIR / f"combo_task3_{out_name}_{setting}.log"
        cmd = build_cmd(mc, setting)
        with open(log_path, "ab") as logf:
            p = subprocess.Popen(
                cmd, env=env,
                stdout=logf, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        spawned.append((out_name, setting, p.pid))
        print(f"  PID {p.pid:>7}  task3 × {out_name:<35} × {setting}")

    print(f"\n总共 spawn {len(spawned)} 个进程, 全部 detached.")
    print(f"进度: tail -f {PROGRESS_LOG}")


if __name__ == "__main__":
    main()
