"""
Run a Qwen3-VL model on the supra_vqa logka eval set: two images per prompt
(host first, guest second), predict a single float logKa.

Run:
    python3 supra_vqa/scripts/run_logka.py
    python3 supra_vqa/scripts/run_logka.py --limit 5

Output (one directory per invocation):
    supra_vqa/results/<model>_logka_<YYYYMMDD-HHMMSS>/
        predictions.csv         pair_id, host_id, guest_id, raw_output, pred
        predictions_scored.csv  per-row gold/pred/abs_error
        summary.json            this run's metrics
        run_info.json           inputs + CLI args + prompt
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

SUPRA_VQA_ROOT = Path(__file__).resolve().parent.parent
EVAL_CSV = SUPRA_VQA_ROOT / "logka.csv"
RESULTS_DIR = SUPRA_VQA_ROOT / "results"
MODEL_PATH = "/home/wsun4/aisci_data/Qwen3-VL-4B-Instruct"

PROMPT = (
    "The first image is a HOST molecule. The second image is its GUEST molecule. "
    "Estimate logKa = log10(Ka) for the 1:1 host-guest complex in water at ~25 C, "
    "where Ka is the association constant in M^-1. "
    "Reply with ONLY the value of logKa as a single decimal number, "
    "typically in the range -3 to 16. "
    "Do NOT output the raw Ka, do NOT use scientific notation, "
    "no units, no explanation, no prefix."
)

FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def postprocess(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    m = FLOAT_RE.search(s)
    return m.group(0) if m else ""


def load_eval_rows(limit: int | None) -> list[dict]:
    with EVAL_CSV.open() as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--model", type=str, default=MODEL_PATH)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    rows = load_eval_rows(args.limit)
    if not rows:
        sys.exit("no rows to run on")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = Path(args.model).name
    run_id = f"{model_name}_logka_{timestamp}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "predictions.csv"
    print(f"run_id: {run_id}")
    print(f"running logka on {len(rows)} rows -> {out_path}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    messages = [{"role": "user", "content": [
        {"type": "image"},  # host
        {"type": "image"},  # guest
        {"type": "text", "text": PROMPT},
    ]}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 2},
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    inputs, kept_rows = [], []
    for r in rows:
        h_path = SUPRA_VQA_ROOT / r["host_image"]
        g_path = SUPRA_VQA_ROOT / r["guest_image"]
        try:
            h_img = Image.open(h_path).convert("RGB")
            g_img = Image.open(g_path).convert("RGB")
        except Exception as e:
            print(f"[skip] {r['pair_id']}: {e}", file=sys.stderr)
            continue
        inputs.append({"prompt": prompt_text,
                       "multi_modal_data": {"image": [h_img, g_img]}})
        kept_rows.append(r)

    outputs = llm.generate(inputs, sampling)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "host_id", "guest_id", "raw_output", "pred"])
        w.writeheader()
        for r, out in zip(kept_rows, outputs):
            raw = out.outputs[0].text if out.outputs else ""
            w.writerow({"pair_id": r["pair_id"], "host_id": r["host_id"],
                        "guest_id": r["guest_id"],
                        "raw_output": raw, "pred": postprocess(raw)})

    run_info = {
        "run_id": run_id,
        "model": model_name,
        "model_path": args.model,
        "mode": "logka",
        "timestamp": timestamp,
        "n": len(kept_rows),
        "limit": args.limit,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_mem,
        "prompt": PROMPT,
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"wrote {out_path}  ({len(kept_rows)} rows)")
    print(f"run_info -> {run_dir / 'run_info.json'}")

    if not args.no_eval:
        print("--- evaluating ---")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from evaluate_logka import score_run
        score_run(run_dir)


if __name__ == "__main__":
    main()
