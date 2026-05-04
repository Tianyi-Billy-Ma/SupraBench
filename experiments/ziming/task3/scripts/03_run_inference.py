"""
Task 1 · Step 03c — vLLM 本地推理（开放问答:host/guest 性质解释）

模型加载一次,顺序跑 base / cot / fewshot 三个 setting,节省 GPU 时间。

特性:
  - 批量推理 (高吞吐)
  - 断点续跑: CSV 行级 append, 按 id 去重
  - 抽取 <answer>...</answer> 内文本作为 pred_answer
  - 完整日志: full_log.jsonl 保留每条 prompt + response

用法:
    # 一次跑全部 3 个 setting (推荐)
    python scripts/03c_run_inference_task1.py --model Qwen/Qwen3-8B --prompt all

    # 单 setting
    python scripts/03c_run_inference_task1.py --model Qwen/Qwen3-8B --prompt base

    # 测试: 只跑前 10 行
    python scripts/03c_run_inference_task1.py --model Qwen/Qwen3-8B --prompt all --limit 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/anvil/projects/x-cis260048/.env")
os.environ.setdefault("HF_HOME", "/anvil/projects/x-cis260048/hf_cache")

from vllm import LLM, SamplingParams  # noqa: E402


ALL_PROMPTS = ["base", "cot", "fewshot"]
MAX_TOKENS_MAP = {"base": 512, "fewshot": 512, "cot": 1024}

ANSWER_TAG_RE = re.compile(
    r"<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>",
    re.IGNORECASE | re.DOTALL,
)


def parse_answer(text: str) -> tuple[str, str]:
    """从模型输出抽取 <answer>...</answer> 文本; 失败时回退到全文。"""
    if not text:
        return "", "parse_fail"
    m = ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).strip(), "ok"
    # 回退: 取最后一段非空文本
    cleaned = text.strip()
    if cleaned:
        return cleaned, "fallback_full"
    return "", "parse_fail"


def model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace("-", "_")


def run_one_setting(
    llm: LLM,
    prompt_type: str,
    model_id: str,
    root: Path,
    limit: int | None,
    temperature: float,
) -> dict:
    data_dir = root / "data" / "task1"
    prompt_file = data_dir / f"prompts_{prompt_type}.jsonl"

    if not prompt_file.exists():
        print(f"  [SKIP] {prompt_file} 不存在")
        return {}

    prompts_data = [json.loads(l) for l in prompt_file.open() if l.strip()]

    slug = model_slug(model_id)
    out_dir = root / "outputs" / f"task1_{slug}_{prompt_type}"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.csv"
    full_log_path = out_dir / "full_log.jsonl"
    meta_path = out_dir / "run_meta.json"

    # 断点续跑
    done_ids: set[str] = set()
    if preds_path.exists() and preds_path.stat().st_size > 0:
        import pandas as pd
        done_df = pd.read_csv(preds_path)
        done_ids = set(done_df["id"].astype(str).tolist())
        print(f"  [resume] 已有 {len(done_ids)} 条, 跳过")

    todo = [p for p in prompts_data if p["id"] not in done_ids]
    if limit:
        todo = todo[:limit]

    if not todo:
        print(f"  [done] 无待推理行")
        return {"prompt_type": prompt_type, "n": 0, "status": "skipped"}

    max_tokens = MAX_TOKENS_MAP.get(prompt_type, 512)
    print(f"  待推理: {len(todo)} 行, max_tokens={max_tokens}")

    meta = {
        "model": model_id,
        "prompt_type": prompt_type,
        "engine": "vllm",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "total_prompts": len(prompts_data),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["</answer>"],
    )

    t0 = time.time()
    prompt_texts = [rec["prompt"] for rec in todo]
    outputs = llm.generate(prompt_texts, sampling_params)
    dt = time.time() - t0
    print(f"  推理完成 ({dt:.1f}s, {len(todo)/dt:.1f} samples/s)")

    is_new = not preds_path.exists() or preds_path.stat().st_size == 0
    n_fail = 0
    cols = [
        "id", "subtype", "host_name", "guest_name", "guest_smiles",
        "gold_answer", "pred_answer", "parse_status", "raw_output",
    ]
    with preds_path.open("a", newline="") as f_csv, \
         full_log_path.open("a") as f_log:
        writer = csv.writer(f_csv)
        if is_new:
            writer.writerow(cols)
        for rec, output in zip(todo, outputs):
            raw = output.outputs[0].text
            raw_for_parse = raw + "</answer>" if "<answer>" in raw and "</answer>" not in raw else raw
            pred, status = parse_answer(raw_for_parse)
            if status == "parse_fail":
                n_fail += 1
            raw_short = raw.replace("\n", " ").strip()[:1000]
            writer.writerow([
                rec["id"], rec["subtype"],
                rec.get("host_name") or "",
                rec.get("guest_name") or "",
                rec.get("guest_smiles") or "",
                rec["answer"], pred, status, raw_short,
            ])
            f_log.write(json.dumps({
                "id": rec["id"],
                "subtype": rec["subtype"],
                "gold_answer": rec["answer"],
                "pred_answer": pred,
                "parse_status": status,
                "prompt": rec["prompt"],
                "response": raw,
            }, ensure_ascii=False) + "\n")

    fail_rate = n_fail / max(1, len(todo))
    result = {
        "prompt_type": prompt_type,
        "n": len(todo),
        "parse_fail_rate": float(fail_rate),
        "time_s": round(dt, 1),
    }
    print(f"  → n={len(todo)}  parse_fail={fail_rate:.1%}  ({dt:.0f}s)")
    return result


def main(args: argparse.Namespace) -> int:
    root = Path(args.root)
    prompt_list = ALL_PROMPTS if args.prompt == "all" else [args.prompt]

    print("=" * 60)
    print("Task 1 vLLM Inference")
    print(f"  Model:    {args.model}")
    print(f"  Settings: {prompt_list}")
    print(f"  Limit:    {args.limit or 'full'}")
    print("=" * 60)

    print(f"\n[load] 加载模型 {args.model} ...")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
    )
    load_time = time.time() - t0
    print(f"[load] 完成 ({load_time:.1f}s)\n")

    results = []
    for pt in prompt_list:
        print(f"\n{'─'*40}")
        print(f"[{pt}] 开始")
        print(f"{'─'*40}")
        r = run_one_setting(llm, pt, args.model, root, args.limit, args.temperature)
        results.append(r)

    print(f"\n{'='*60}")
    print(f"汇总 ({args.model})")
    print(f"{'='*60}")
    print(f"模型加载: {load_time:.0f}s")
    print(f"{'Setting':<12} {'N':>6} {'Fail%':>8} {'Time':>8}")
    print("-" * 40)
    for r in results:
        if r.get("status") == "skipped":
            print(f"{r['prompt_type']:<12} {'skipped':>6}")
        else:
            print(
                f"{r['prompt_type']:<12} {r['n']:>6} "
                f"{r.get('parse_fail_rate',0):>7.1%} "
                f"{r.get('time_s',0):>7.0f}s"
            )
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--model", required=True, help="HuggingFace model ID")
    ap.add_argument("--prompt", default="all",
                    choices=["base", "cot", "fewshot", "all"])
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="每个 setting 只跑前 N 行 (测试用)")
    args = ap.parse_args()
    sys.exit(main(args))
