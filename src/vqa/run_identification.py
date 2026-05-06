"""
Run a Qwen3-VL model on the supra_vqa identification eval set via vLLM.

Modes:
    name    — ask for the molecule name
    smiles  — ask for a canonical SMILES string

Run:
    python3 src/vqa/run_identification.py --mode name
    python3 src/vqa/run_identification.py --mode smiles --limit 5
    python3 src/vqa/run_identification.py --mode name --model /path/to/model

Output (one directory per invocation):
    supra-vqa/results/<model>_<mode>_<YYYYMMDD-HHMMSS>/
        predictions.csv         molecule_id, image, raw_output, pred
        predictions_scored.csv  + per-row metrics (auto-eval)
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

SUPRA_VQA_ROOT = Path(__file__).resolve().parents[2] / "supra-vqa"
EVAL_CSV = SUPRA_VQA_ROOT / "identification.csv"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
MODEL_PATH = "/home/wsun4/aisci_data/Qwen3-VL-4B-Instruct"

PROMPTS = {
    "name": (
        "You are shown the 2D structural drawing of a single small molecule. "
        "Reply with ONLY the molecule's common name or accepted chemical name. "
        "No explanation, no prefix, no quotation marks."
    ),
    "smiles": (
        "You are shown the 2D structural drawing of a single small molecule. "
        "Reply with ONLY the SMILES string for this molecule. "
        "Output one line containing only the SMILES. No explanation, no prefix, "
        "no backticks, no code fences."
    ),
}


def postprocess_name(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    s = s.splitlines()[0].strip().strip('"').strip("'").strip("`")
    if s.endswith("."):
        s = s[:-1].strip()
    return s


SMILES_TOKEN_RE = re.compile(r"\S+")


def postprocess_smiles(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    s = s.replace("```", " ").replace("`", " ")
    s = re.sub(r"(?i)^\s*smiles\s*[:=]\s*", "", s)
    line = s.splitlines()[0].strip()
    m = SMILES_TOKEN_RE.search(line)
    return m.group(0) if m else ""


def load_eval_rows(limit: int | None) -> list[dict]:
    with EVAL_CSV.open() as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["name", "smiles"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smiles-only", action="store_true",
                    help="in name mode, also drop rows that lack cano_smiles "
                         "(matches the smiles-mode sample for direct comparison)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--model", type=str, default=MODEL_PATH)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    rows = load_eval_rows(args.limit)
    if args.mode == "smiles" or args.smiles_only:
        rows = [r for r in rows if r["cano_smiles"]]
    if not rows:
        sys.exit("no rows to run on")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = Path(args.model).name
    run_id = f"{model_name}_{args.mode}_{timestamp}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "predictions.csv"
    print(f"run_id: {run_id}")
    print(f"running {args.mode} on {len(rows)} rows -> {out_path}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": PROMPTS[args.mode]},
    ]}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    inputs, kept_rows = [], []
    for r in rows:
        path = SUPRA_VQA_ROOT / r["image"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[skip] {path}: {e}", file=sys.stderr)
            continue
        inputs.append({"prompt": prompt_text, "multi_modal_data": {"image": img}})
        kept_rows.append(r)

    outputs = llm.generate(inputs, sampling)

    postproc = postprocess_name if args.mode == "name" else postprocess_smiles
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["molecule_id", "image", "raw_output", "pred"])
        w.writeheader()
        for r, out in zip(kept_rows, outputs):
            raw = out.outputs[0].text if out.outputs else ""
            w.writerow({"molecule_id": r["molecule_id"], "image": r["image"],
                        "raw_output": raw, "pred": postproc(raw)})

    run_info = {
        "run_id": run_id,
        "model": model_name,
        "model_path": args.model,
        "mode": args.mode,
        "timestamp": timestamp,
        "n": len(kept_rows),
        "limit": args.limit,
        "smiles_only": args.smiles_only,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_mem,
        "prompt": PROMPTS[args.mode],
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"wrote {out_path}  ({len(kept_rows)} rows)")
    print(f"run_info -> {run_dir / 'run_info.json'}")

    if not args.no_eval:
        print("--- evaluating ---")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from evaluate_identification import score_run
        score_run(run_dir)


if __name__ == "__main__":
    main()
