"""
Task 1 + Task 3 · Step 03d — OpenRouter API 推理 (chat completions)

通过 OpenRouter 调任意 chat 模型 (e.g. anthropic/claude-sonnet-4.6)。
复用 task1 / task3 已有的 prompts_*.jsonl,输出到:
    outputs/task{1,3}/{out_name}/{setting}/predictions.csv

特性:
  - 并发 (ThreadPoolExecutor)
  - 指数退避重试 (429 / 5xx / 网络错误)
  - 断点续跑: 按 id (task1) / interaction_id (task3) 去重
  - 支持 reasoning effort (low/medium/high/minimal) + 关闭 thinking
  - 实验进度 JSONL 日志, 每个 setting 完成后追加一行

用法:
    export OPENROUTER_API_KEY=sk-or-v1-...
    python scripts/03d_run_inference_openrouter.py \
        --task task1 --model anthropic/claude-sonnet-4.6 --prompt all --limit 5

    # OpenAI o-series 高 reasoning
    python scripts/03d_run_inference_openrouter.py \
        --task task3 --model openai/gpt-5.5 --reasoning-effort high \
        --out-name gpt-5.5_xhigh --limit 5

    # 关闭 thinking
    python scripts/03d_run_inference_openrouter.py \
        --task task3 --model google/gemini-3-flash-preview --no-reasoning \
        --out-name gemini-3-flash-preview_nothinking --limit 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
BACKENDS = ("openrouter", "openai")

ALL_PROMPTS = ["base", "fewshot", "cot"]

# task1: 开放问答, 取 <answer>...</answer> 内文本
TASK1_ANSWER_RE = re.compile(
    r"<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>",
    re.IGNORECASE | re.DOTALL,
)
TASK1_MAX_TOKENS = {"base": 4096, "fewshot": 4096, "cot": 8192}

# task3: 6 选 1 多选, 取 <answer>X</answer> 字母
LABELS_ORDER = ["water", "DMSO", "MeCN", "MeOH", "CHCl3", "CH2Cl2"]
LETTER_TO_LABEL = {chr(ord("A") + i): lab for i, lab in enumerate(LABELS_ORDER)}
LABEL_TO_LETTER = {v: k for k, v in LETTER_TO_LABEL.items()}
TASK3_ANSWER_TAG_RE = re.compile(r"<\s*answer\s*>\s*([A-F])\s*<\s*/\s*answer\s*>", re.IGNORECASE)
TASK3_LONE_LETTER_RE = re.compile(r"\b([A-F])\b")
TASK3_LABEL_LITERAL_RE = re.compile(
    r"\b(" + "|".join(re.escape(L) for L in LABELS_ORDER) + r")\b", re.IGNORECASE
)
TASK3_LABEL_NORM = {L.lower(): L for L in LABELS_ORDER}
TASK3_MAX_TOKENS = {"base": 4096, "fewshot": 4096, "cot": 8192}


def macro_f1(y_true, y_pred, labels):
    per = {}
    f1s = []
    for L in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == L and p == L)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != L and p == L)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == L and p != L)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        sup = sum(1 for t in y_true if t == L)
        per[L] = {"prec": prec, "rec": rec, "f1": f1, "sup": sup}
        if sup > 0:
            f1s.append(f1)
    return (sum(f1s) / len(f1s) if f1s else 0.0), per


def weighted_f1(per):
    tot = sum(p["sup"] for p in per.values())
    return sum(p["f1"] * p["sup"] for p in per.values()) / tot if tot else 0.0


def parse_task1(text: str) -> tuple[str, str]:
    if not text:
        return "", "parse_fail"
    m = TASK1_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip(), "ok"
    cleaned = text.strip()
    if cleaned:
        return cleaned, "fallback_full"
    return "", "parse_fail"


def parse_task3(text: str) -> tuple[str, str]:
    if not text:
        return "?", "parse_fail"
    m = TASK3_ANSWER_TAG_RE.search(text)
    if m:
        letter = m.group(1).upper()
        return letter, LETTER_TO_LABEL[letter]
    matches = TASK3_LONE_LETTER_RE.findall(text)
    if matches:
        letter = matches[-1].upper()
        return letter, LETTER_TO_LABEL[letter]
    m = TASK3_LABEL_LITERAL_RE.search(text)
    if m:
        label = TASK3_LABEL_NORM[m.group(1).lower()]
        return LABEL_TO_LETTER[label], label
    return "?", "parse_fail"


def default_out_name(model_id: str) -> str:
    """anthropic/claude-sonnet-4.6 → claude-sonnet-4.6 (lowercase)"""
    return model_id.split("/")[-1].lower()


def _openai_reasoning_effort(reasoning: dict | None) -> str | None:
    """把 OpenRouter 风格 reasoning dict 翻译成 OpenAI 的 reasoning_effort 字符串。
    OpenAI gpt-5.x 支持: none / low / medium / high / xhigh (不支持 'minimal').
    {"effort": "high"} → "high"
    {"exclude": True, "enabled": False} → "none" (彻底关 thinking)
    {"effort": "minimal"} → "none" (OpenRouter 的 minimal 在 OpenAI 上对应 none)
    None / 其他 → None (不发该字段, 用模型默认)
    """
    if not reasoning:
        return None
    if reasoning.get("exclude") or reasoning.get("enabled") is False:
        return "none"
    eff = reasoning.get("effort")
    if eff == "minimal":
        return "none"
    if eff in ("none", "low", "medium", "high", "xhigh"):
        return eff
    return None


def call_openrouter(
    prompt: str,
    model: str,
    api_key: str,
    *,
    backend: str = "openrouter",
    temperature: float = 0.0,
    max_tokens: int = 512,
    reasoning: dict | None = None,
    timeout: int = 180,
    max_retries: int = 5,
) -> tuple[str, str | None]:
    """统一 chat completions caller (OpenRouter / OpenAI 直连)。
    返回 (raw_text, error_or_None)。错误时 raw_text 以 '[ERROR]' 开头。"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if backend == "openai":
        url = OPENAI_URL
        # OpenAI 平台 model id 不带 vendor 前缀
        api_model = model.split("/", 1)[-1] if "/" in model else model
        eff = _openai_reasoning_effort(reasoning)
        # gpt-5.x xhigh/high reasoning 会吃掉大量 token (10K-30K), 自动放大上限
        # 否则 reasoning 用满 → content 是空 → parse_fail
        if eff in ("high", "xhigh"):
            api_max_tokens = max(max_tokens, 32768)
        elif eff in ("medium",):
            api_max_tokens = max(max_tokens, 16384)
        else:
            api_max_tokens = max_tokens
        payload = {
            "model": api_model,
            "messages": [{"role": "user", "content": prompt}],
            # gpt-5.x reasoning 模型必须用 max_completion_tokens
            "max_completion_tokens": api_max_tokens,
            "stream": False,
        }
        # reasoning 模型 (gpt-5.x) 不允许 temperature ≠ 1.0, 干脆省掉这个字段
        if eff is not None:
            payload["reasoning_effort"] = eff
    else:  # openrouter
        url = OPENROUTER_URL
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if reasoning is not None:
            payload["reasoning"] = reasoning

    last_err: str | None = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                try:
                    msg = data["choices"][0]["message"]
                    content = msg.get("content")
                    # Reasoning models (qwen3.5, gpt-5_xhigh, gemini_high 等) 当 finish=length 时
                    # content 可能 null/空, 真正的输出在 reasoning 字段里
                    if not content:
                        content = msg.get("reasoning") or ""
                except (KeyError, IndexError, TypeError):
                    return f"[ERROR] malformed response: {str(data)[:300]}", "malformed"
                return content, None
            if r.status_code == 429:
                # 优先用 server 给的 retry-after, fallback 指数退避
                retry_after = r.headers.get("retry-after") or r.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = max(1.0, float(retry_after))
                    except ValueError:
                        sleep_s = 60.0
                else:
                    sleep_s = min(2 ** attempt + 1, 60)
                last_err = f"HTTP 429 (sleep {sleep_s:.1f}s): {r.text[:200]}"
                time.sleep(sleep_s)
                continue
            if r.status_code in (408, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(min(2 ** attempt, 30))
                continue
            return f"[ERROR] HTTP {r.status_code}: {r.text[:300]}", f"http_{r.status_code}"
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 30))
    return f"[ERROR] retries exhausted: {last_err}", "retry_exhausted"


def append_progress_log(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Task 1 (开放问答)
# ---------------------------------------------------------------------------

def run_task1_setting(prompt_type, model, out_name, api_key, root, limit,
                      temperature, concurrency, reasoning, log_path,
                      backend="openrouter"):
    data_dir = root / "data" / "task1"
    prompt_file = data_dir / f"prompts_{prompt_type}.jsonl"
    if not prompt_file.exists():
        print(f"  [SKIP] {prompt_file} 不存在")
        return {}

    prompts_data = [json.loads(l) for l in prompt_file.open() if l.strip()]
    out_dir = root / "outputs" / "task1" / out_name / prompt_type
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.csv"
    full_log_path = out_dir / "full_log.jsonl"
    meta_path = out_dir / "run_meta.json"

    done_ids: set[str] = set()
    if preds_path.exists() and preds_path.stat().st_size > 0:
        df = pd.read_csv(preds_path)
        done_ids = set(df["id"].astype(str).tolist())
        print(f"  [resume] 已有 {len(done_ids)} 条, 跳过")

    todo = [p for p in prompts_data if p["id"] not in done_ids]
    if limit:
        todo = todo[:limit]
    if not todo:
        print(f"  [done] 无待推理行")
        return {"prompt_type": prompt_type, "n": 0, "status": "skipped"}

    max_tokens = TASK1_MAX_TOKENS.get(prompt_type, 512)
    print(f"  待推理: {len(todo)} 行, max_tokens={max_tokens}, concurrency={concurrency}")

    meta = {
        "model": model, "out_name": out_name, "prompt_type": prompt_type,
        "engine": backend,
        "endpoint": OPENAI_URL if backend == "openai" else OPENROUTER_URL,
        "max_tokens": max_tokens, "temperature": temperature,
        "concurrency": concurrency, "reasoning": reasoning,
        "total_prompts": len(prompts_data), "limit": limit,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    is_new = not preds_path.exists() or preds_path.stat().st_size == 0
    cols = [
        "id", "subtype", "host_name", "guest_name", "guest_smiles",
        "gold_answer", "pred_answer", "parse_status", "raw_output",
    ]

    t0 = time.time()
    n_fail = 0
    n_done = 0
    with preds_path.open("a", newline="") as f_csv, full_log_path.open("a") as f_log:
        writer = csv.writer(f_csv)
        if is_new:
            writer.writerow(cols)
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {
                ex.submit(call_openrouter, rec["prompt"], model, api_key,
                          backend=backend,
                          temperature=temperature, max_tokens=max_tokens,
                          reasoning=reasoning): rec
                for rec in todo
            }
            for fut in as_completed(futs):
                rec = futs[fut]
                raw, err = fut.result()
                pred, status = parse_task1(raw)
                if status == "parse_fail" or err:
                    n_fail += 1
                raw_short = raw.replace("\n", " ").strip()[:1000]
                writer.writerow([
                    rec["id"], rec["subtype"],
                    rec.get("host_name") or "",
                    rec.get("guest_name") or "",
                    rec.get("guest_smiles") or "",
                    rec["answer"], pred, status, raw_short,
                ])
                f_csv.flush()
                f_log.write(json.dumps({
                    "id": rec["id"], "subtype": rec["subtype"],
                    "gold_answer": rec["answer"], "pred_answer": pred,
                    "parse_status": status, "error": err,
                    "prompt": rec["prompt"], "response": raw,
                }, ensure_ascii=False) + "\n")
                f_log.flush()
                n_done += 1
                if n_done % 20 == 0 or n_done == len(todo):
                    dt = time.time() - t0
                    rps = n_done / dt if dt else 0
                    eta = (len(todo) - n_done) / rps if rps else 0
                    print(f"  [{n_done}/{len(todo)}] fail={n_fail} ({n_fail/n_done:.1%}) "
                          f"{rps:.2f} req/s  ETA {eta/60:.1f}m")

    dt = time.time() - t0
    fail_rate = n_fail / max(1, len(todo))
    result = {"prompt_type": prompt_type, "n": len(todo),
              "parse_fail_rate": float(fail_rate), "time_s": round(dt, 1)}
    print(f"  → n={len(todo)}  parse_fail={fail_rate:.1%}  ({dt:.0f}s)")

    if log_path:
        append_progress_log(log_path, {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task": "task1", "model": model, "out_name": out_name,
            "prompt_type": prompt_type, "n": len(todo),
            "parse_fail_rate": float(fail_rate),
            "time_s": round(dt, 1), "reasoning": reasoning,
            "status": "ok",
        })
    return result


# ---------------------------------------------------------------------------
# Task 3 (6 选 1)
# ---------------------------------------------------------------------------

def run_task3_setting(prompt_type, model, out_name, api_key, root, limit,
                      temperature, concurrency, reasoning, log_path,
                      backend="openrouter"):
    data_dir = root / "data" / "task7"
    prompt_file = data_dir / f"prompts_{prompt_type}.jsonl"
    if not prompt_file.exists():
        print(f"  [SKIP] {prompt_file} 不存在")
        return {}

    prompts_data = [json.loads(l) for l in prompt_file.open() if l.strip()]
    out_dir = root / "outputs" / "task3" / out_name / prompt_type
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.csv"
    full_log_path = out_dir / "full_log.jsonl"
    meta_path = out_dir / "run_meta.json"

    done_ids: set[int] = set()
    if preds_path.exists() and preds_path.stat().st_size > 0:
        df = pd.read_csv(preds_path)
        done_ids = set(df["interaction_id"].astype(int).tolist())
        print(f"  [resume] 已有 {len(done_ids)} 条, 跳过")

    todo = [p for p in prompts_data if p["interaction_id"] not in done_ids]
    if limit:
        todo = todo[:limit]
    if not todo:
        print(f"  [done] 无待推理行")
        return {"prompt_type": prompt_type, "n": 0, "status": "skipped"}

    max_tokens = TASK3_MAX_TOKENS.get(prompt_type, 512)
    print(f"  待推理: {len(todo)} 行, max_tokens={max_tokens}, concurrency={concurrency}")

    meta = {
        "model": model, "out_name": out_name, "prompt_type": prompt_type,
        "engine": backend,
        "endpoint": OPENAI_URL if backend == "openai" else OPENROUTER_URL,
        "max_tokens": max_tokens, "temperature": temperature,
        "concurrency": concurrency, "reasoning": reasoning,
        "total_prompts": len(prompts_data), "limit": limit,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    is_new = not preds_path.exists() or preds_path.stat().st_size == 0
    cols = ["interaction_id", "true_label", "true_letter",
            "pred_letter", "pred_label", "raw_output"]

    t0 = time.time()
    n_fail = 0
    n_done = 0
    with preds_path.open("a", newline="") as f_csv, full_log_path.open("a") as f_log:
        writer = csv.writer(f_csv)
        if is_new:
            writer.writerow(cols)
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {
                ex.submit(call_openrouter, rec["prompt"], model, api_key,
                          backend=backend,
                          temperature=temperature, max_tokens=max_tokens,
                          reasoning=reasoning): rec
                for rec in todo
            }
            for fut in as_completed(futs):
                rec = futs[fut]
                raw, err = fut.result()
                letter, label = parse_task3(raw)
                if label == "parse_fail" or err:
                    n_fail += 1
                raw_short = raw.replace("\n", " ").strip()[:500]
                writer.writerow([
                    rec["interaction_id"], rec["true_label"], rec["true_letter"],
                    letter, label, raw_short,
                ])
                f_csv.flush()
                f_log.write(json.dumps({
                    "interaction_id": rec["interaction_id"],
                    "true_label": rec["true_label"], "true_letter": rec["true_letter"],
                    "pred_letter": letter, "pred_label": label, "error": err,
                    "prompt": rec["prompt"], "response": raw,
                }, ensure_ascii=False) + "\n")
                f_log.flush()
                n_done += 1
                if n_done % 50 == 0 or n_done == len(todo):
                    dt = time.time() - t0
                    rps = n_done / dt if dt else 0
                    eta = (len(todo) - n_done) / rps if rps else 0
                    print(f"  [{n_done}/{len(todo)}] fail={n_fail} "
                          f"{rps:.2f} req/s  ETA {eta/60:.1f}m")

    dt = time.time() - t0
    fail_rate = n_fail / max(1, len(todo))

    df_all = pd.read_csv(preds_path)
    valid_df = df_all[df_all["pred_label"] != "parse_fail"]
    acc = (valid_df["pred_label"] == valid_df["true_label"]).mean() if len(valid_df) else 0.0
    mf1, per_class = macro_f1(
        valid_df["true_label"].tolist(), valid_df["pred_label"].tolist(), LABELS_ORDER
    )
    wf1 = weighted_f1(per_class)
    print(f"  → acc={acc:.3f}  macro_F1={mf1:.3f}  weighted_F1={wf1:.3f}  "
          f"parse_fail={fail_rate:.1%}  ({dt:.0f}s)")
    result = {"prompt_type": prompt_type, "n": len(todo),
              "parse_fail_rate": float(fail_rate),
              "accuracy": float(acc),
              "macro_f1": float(mf1),
              "weighted_f1": float(wf1),
              "per_class": per_class,
              "time_s": round(dt, 1)}

    if log_path:
        append_progress_log(log_path, {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task": "task3", "model": model, "out_name": out_name,
            "prompt_type": prompt_type, "n": len(todo),
            "parse_fail_rate": float(fail_rate),
            "accuracy": float(acc), "macro_f1": float(mf1),
            "weighted_f1": float(wf1),
            "time_s": round(dt, 1), "reasoning": reasoning,
            "status": "ok",
        })
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if args.backend == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[ERROR] --backend openai 但未设置 OPENAI_API_KEY (或 --api-key)",
                  file=sys.stderr)
            return 2
    else:
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("[ERROR] 未设置 OPENROUTER_API_KEY (或 --api-key)", file=sys.stderr)
            return 2

    # reasoning payload 构造
    reasoning = None
    if args.no_reasoning:
        reasoning = {"exclude": True, "enabled": False}
    elif args.reasoning_effort:
        reasoning = {"effort": args.reasoning_effort}
    elif args.reasoning_json:
        reasoning = json.loads(args.reasoning_json)

    out_name = args.out_name or default_out_name(args.model)
    log_path = Path(args.log_path) if args.log_path else None

    root = Path(args.root)
    prompt_list = ALL_PROMPTS if args.prompt == "all" else [args.prompt]

    print("=" * 60)
    print(f"Inference ({args.backend}) - {args.task}")
    print(f"  Model:    {args.model}")
    print(f"  OutName:  {out_name}")
    print(f"  Settings: {prompt_list}")
    print(f"  Limit:    {args.limit or 'full'}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Reasoning: {reasoning}")
    print(f"  ProgLog:  {log_path}")
    print("=" * 60)

    runner = {"task1": run_task1_setting, "task3": run_task3_setting}[args.task]

    results = []
    for pt in prompt_list:
        print(f"\n{'─'*40}\n[{pt}] 开始\n{'─'*40}")
        try:
            r = runner(pt, args.model, out_name, api_key, root, args.limit,
                       args.temperature, args.concurrency, reasoning, log_path,
                       backend=args.backend)
            results.append(r)
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f"  [SETTING FAILED] {err_msg}")
            if log_path:
                append_progress_log(log_path, {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "task": args.task, "model": args.model, "out_name": out_name,
                    "prompt_type": pt, "status": "exception", "error": err_msg,
                })
            results.append({"prompt_type": pt, "status": "exception", "error": err_msg})

    print(f"\n{'='*60}\n汇总 ({args.model}, {args.task})\n{'='*60}")
    if args.task == "task3":
        print(f"{'Setting':<12} {'N':>5} {'Acc':>7} {'macroF1':>9} {'wF1':>7} {'Fail%':>7} {'Time':>7}")
    else:
        print(f"{'Setting':<12} {'N':>6} {'Fail%':>8} {'Time':>8}")
    print("-" * 60)
    for r in results:
        if not r:
            continue
        if r.get("status") == "skipped":
            print(f"{r['prompt_type']:<12} {'skipped':>6}")
        elif r.get("status") == "exception":
            print(f"{r['prompt_type']:<12} EXCEPTION {r.get('error','')[:50]}")
        elif args.task == "task3":
            print(f"{r['prompt_type']:<12} {r['n']:>5} {r.get('accuracy',0):>7.3f} "
                  f"{r.get('macro_f1',0):>9.3f} {r.get('weighted_f1',0):>7.3f} "
                  f"{r.get('parse_fail_rate',0):>6.1%} {r.get('time_s',0):>6.0f}s")
        else:
            print(f"{r['prompt_type']:<12} {r['n']:>6} "
                  f"{r.get('parse_fail_rate',0):>7.1%} {r.get('time_s',0):>7.0f}s")

    if args.task == "task3":
        print(f"\n--- per-class F1 ---")
        for r in results:
            if not r or "per_class" not in r:
                continue
            print(f"\n[{r['prompt_type']}]")
            for L, p in r["per_class"].items():
                if p["sup"] > 0:
                    print(f"  {L:<8} sup={p['sup']:>4} prec={p['prec']:.3f} "
                          f"rec={p['rec']:.3f} f1={p['f1']:.3f}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--task", required=True, choices=["task1", "task3"])
    ap.add_argument("--backend", default="openrouter", choices=list(BACKENDS),
                    help="API backend. 'openai' uses OPENAI_API_KEY + "
                         "max_completion_tokens; 'openrouter' is default.")
    ap.add_argument("--model", required=True,
                    help="OpenRouter model id (e.g. anthropic/claude-sonnet-4.6) "
                         "or OpenAI model id (e.g. gpt-5.4-mini). For openai backend "
                         "any 'vendor/' prefix is stripped.")
    ap.add_argument("--out-name", default=None,
                    help="Override 输出目录名 (default: model id 末段)")
    ap.add_argument("--prompt", default="all",
                    choices=["base", "fewshot", "cot", "all"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["minimal", "low", "medium", "high", "xhigh", "none"],
                    help="reasoning effort. OpenRouter accepts minimal/low/medium/high; "
                         "OpenAI gpt-5.x accepts none/low/medium/high/xhigh "
                         "(translation handled internally).")
    ap.add_argument("--no-reasoning", action="store_true",
                    help="禁用 thinking (设 reasoning.exclude=true, enabled=false)")
    ap.add_argument("--reasoning-json", default=None,
                    help="自定义 reasoning JSON, e.g. '{\"max_tokens\": 8000}'")
    ap.add_argument("--log-path", default=None,
                    help="实验进度 JSONL 日志, append 一条 / setting")
    args = ap.parse_args()
    sys.exit(main(args))
