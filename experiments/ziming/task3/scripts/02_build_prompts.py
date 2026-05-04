"""
Task 1 · Step 02c — Prompt 渲染（开放问答:host/guest 性质解释）

源数据:
  data/task1/task3.jsonl    134 行, 两种 subtype (forward 41 / reverse 93)

输出（写入 data/task1/）:
  prompts_base.jsonl       零样本 + 无 CoT
  prompts_cot.jsonl        零样本 + vanilla CoT ("Let's think step by step.")
  prompts_fewshot.jsonl    6-shot + 无 CoT
  fewshot_examples.json    挑出的示例清单（forward 6 + reverse 6）

设计:
  - 每行的 question 字段已是完整 prompt, 抽取中间 "Question: ..." 段,
    再走 SupraBench/src/templates 的 generate_prompt 统一包装,
    保证 base/cot/fewshot 的头尾完全一致。
  - Few-shot 按 subtype 匹配: forward 测试用 forward 示例, reverse 用 reverse,
    避免方向错位。每 subtype 取 id 排序前 6 条, 从评估集剔除以防泄漏。

用法:
    python scripts/02c_build_prompts_task1.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "SupraBench" / "src"))
from templates import generate_prompt  # noqa: E402


K_FEWSHOT = 6
SUBTYPES = ("forward", "reverse")

QUESTION_PREFIX = (
    "You are an expert in supramolecular chemistry.\n"
    "Your task is to answer the following question.\n"
    "Question: "
)
ANSWER_SUFFIX = "Put your final answer between <answer></answer>"


def extract_inner_question(full_question: str) -> str:
    """从源 question 字段抽取 'Question: ' 与 'Put your final answer' 之间的内容。"""
    s = full_question
    if s.startswith(QUESTION_PREFIX):
        s = s[len(QUESTION_PREFIX):]
    idx = s.rfind(ANSWER_SUFFIX)
    if idx != -1:
        s = s[:idx]
    return s.strip()


def build_meta(row: dict) -> dict:
    """从源行抽取元信息字段（缺失则 None）。"""
    return {
        "host_name": row.get("host_name"),
        "guest_name": row.get("guest_name"),
        "guest_smiles": row.get("guest_smiles"),
        # forward 专有数值字段（reverse 行为 None）
        "n_guests_smi": row.get("n_guests_smi"),
        "n_top": row.get("n_top"),
        "gt_mw_mean": row.get("gt_mw_mean"),
        "gt_mw_std": row.get("gt_mw_std"),
        "gt_charge": row.get("gt_charge"),
        "gt_rings_mean": row.get("gt_rings_mean"),
        # reverse 专有
        "n_hosts": row.get("n_hosts"),
        "n_top_hosts": row.get("n_top_hosts"),
        "max_logka": row.get("max_logka"),
    }


def pick_fewshot(rows_by_subtype: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """每个 subtype 按 id 稳定排序取前 K 条。"""
    picked = {}
    for st, rows in rows_by_subtype.items():
        sorted_rows = sorted(rows, key=lambda r: r["id"])
        picked[st] = sorted_rows[:K_FEWSHOT]
    return picked


def render_prompt(
    inner_q: str,
    fewshot_rows: list[dict] | None,
    thinking: bool,
) -> str:
    fewshot_examples = None
    if fewshot_rows:
        fewshot_examples = [
            {
                "query": extract_inner_question(ex["question"]),
                "answer": ex["answer"],
            }
            for ex in fewshot_rows
        ]
    return generate_prompt(inner_q, fewshot_examples=fewshot_examples, thinking=thinking)


def build_one(
    eval_rows: list[dict],
    fewshot_by_st: dict[str, list[dict]],
    out_path: Path,
    *,
    use_fewshot: bool,
    use_cot: bool,
) -> tuple[int, list[int]]:
    n = 0
    lengths: list[int] = []
    with out_path.open("w") as f:
        for row in eval_rows:
            inner_q = extract_inner_question(row["question"])
            shots = fewshot_by_st[row["subtype"]] if use_fewshot else None
            prompt = render_prompt(inner_q, shots, thinking=use_cot)
            rec = {
                "id": row["id"],
                "subtype": row["subtype"],
                **build_meta(row),
                "answer": row["answer"],
                "prompt": prompt,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            lengths.append(len(prompt))
            n += 1
    return n, lengths


def main(root: Path) -> int:
    src = root / "data" / "task1" / "task3.jsonl"
    out_dir = root / "data" / "task1"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in src.open()]
    by_st = {st: [r for r in rows if r["subtype"] == st] for st in SUBTYPES}
    print(f"[load] {len(rows)} 行 ← {src}")
    for st in SUBTYPES:
        print(f"  {st}: {len(by_st[st])}")

    fewshot_by_st = pick_fewshot(by_st)
    fewshot_ids = {r["id"] for st in SUBTYPES for r in fewshot_by_st[st]}
    eval_rows = [r for r in rows if r["id"] not in fewshot_ids]
    print(f"[fewshot] 每 subtype 取 {K_FEWSHOT} 条, 共 {len(fewshot_ids)} 条剔除")
    print(f"[eval] 剩余 {len(eval_rows)} 行用于评估")

    # 写出 few-shot 清单
    ex_meta = out_dir / "fewshot_examples.json"
    ex_meta.write_text(
        json.dumps(
            {
                st: [
                    {
                        "id": r["id"],
                        "subtype": st,
                        "host_name": r.get("host_name"),
                        "guest_name": r.get("guest_name"),
                        "guest_smiles": r.get("guest_smiles"),
                        "answer": r["answer"],
                    }
                    for r in fewshot_by_st[st]
                ]
                for st in SUBTYPES
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"[fewshot] 示例清单 → {ex_meta}")

    configs = [
        ("prompts_base.jsonl",    False, False),
        ("prompts_cot.jsonl",     False, True),
        ("prompts_fewshot.jsonl", True,  False),
    ]
    for name, use_fs, use_cot in configs:
        out_path = out_dir / name
        n, lengths = build_one(
            eval_rows, fewshot_by_st, out_path,
            use_fewshot=use_fs, use_cot=use_cot,
        )
        med = sorted(lengths)[n // 2]
        print(
            f"  → {name}: {n} 行, "
            f"prompt 字符 min={min(lengths)} med={med} max={max(lengths)}"
        )

    # 抽检：对每种文件取首行打印 prompt 头尾各 600 字符
    print("\n=== 抽检 ===")
    for name, _, _ in configs:
        p = out_dir / name
        rec = json.loads(next(p.open()))
        print(f"\n--- {name} (id={rec['id']}, subtype={rec['subtype']}) ---")
        prompt = rec["prompt"]
        if len(prompt) > 1200:
            print(prompt[:600])
            print("\n... [中间省略] ...\n")
            print(prompt[-600:])
        else:
            print(prompt)
        print("-" * 60)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(_REPO_ROOT))
    args = ap.parse_args()
    sys.exit(main(Path(args.root)))
