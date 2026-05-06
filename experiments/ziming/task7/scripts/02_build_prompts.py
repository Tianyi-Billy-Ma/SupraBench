"""
Task 3 · Step 02b — Prompt 渲染（Host SMILES-only 溶剂预测）

与 Task 2 同一任务，但 host 输入只给 SMILES，不给名字/family。
测试 LLM 能否从纯分子结构推理出溶剂环境。

输入组合（combo 轴）重新设计:
  G: host SMILES + guest 名字                      (最小 baseline)
  H: host SMILES + guest 名字 + guest SMILES + tags (给 guest 结构)
  I: host SMILES + guest SMILES + guest tags        (纯结构,无任何名字)

其他轴与 Task 2 相同:
  - reasoning: none / cot_vanilla / cot_structured
  - shot:      zero / few

输出 JSONL 每行: {interaction_id, true_label, true_letter, prompt}

用法:
    python scripts/02b_build_prompts_task3.py --combo G --reasoning none --shot zero
    python scripts/02b_build_prompts_task3.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "SupraBench" / "src"))
from templates import generate_options, generate_prompt  # noqa: E402


# ---------------------------------------------------------------------------
# Label set — 与 Task 2 一致
# ---------------------------------------------------------------------------

LABELS_ORDER = ["water", "DMSO", "MeCN", "MeOH", "CHCl3", "CH2Cl2"]
LETTER_MAP: dict[str, str] = {lab: chr(ord("A") + i) for i, lab in enumerate(LABELS_ORDER)}
OPTIONS_BLOCK = generate_options(LABELS_ORDER)

COMBOS = ["G", "H", "I"]
REASONINGS = ["none", "cot_vanilla", "cot_structured"]
SHOTS = ["zero", "few"]


# ---------------------------------------------------------------------------
# Guidance — 不再提及 host family
# ---------------------------------------------------------------------------

GUIDANCE = """\
Given a host-guest complex, predict which solvent environment is most appropriate \
for measuring its binding constant. You are given the host molecule as a SMILES string. \
Analyze its structure (cavity size, functional groups, charge, hydrophobicity) to determine \
the solvent class.

General principles:
  - Large hydrophobic cavities with polar portals (e.g. glycoluril-based, sulfonated) → water
  - Neutral macrocycles with aromatic walls, no charged groups → CHCl3 or CH2Cl2
  - Polyether / aza-crown scaffolds → MeOH or MeCN
  - Hydrogen-bond donors/acceptors without water solubility → DMSO"""

STRUCTURED_REASONING = """\
Reason in this order before giving your final answer:
  1. Parse the host SMILES: identify the macrocyclic scaffold type, \
key functional groups, and charge state.
  2. Assess water solubility: charged groups (sulfonate, carboxylate, ammonium) → likely water. \
Neutral hydrophobic → organic solvent.
  3. Identify dominant host-guest interaction from host cavity and guest properties.
  4. Match to solvent: hydrophobic encapsulation → water; \
H-bond driven → CHCl3/CH2Cl2; ion pairing in organic → MeCN/MeOH/DMSO.
  5. Output exactly one letter from A-F."""


# ---------------------------------------------------------------------------
# 字段格式化
# ---------------------------------------------------------------------------


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float) and pd.isna(v):
        return "N/A"
    s = str(v).strip()
    return s if s else "N/A"


def _render_fields(row: pd.Series, combo: str) -> str:
    """
    G: host SMILES + guest name
    H: host SMILES + guest name + guest SMILES + guest tags
    I: host SMILES + guest SMILES + guest tags  (纯结构,无名字)
    """
    lines = [f"Host SMILES: {_fmt(row['host_smiles'])}"]
    if combo in ("G", "H"):
        lines.append(f"Guest name: {_fmt(row['guest'])}")
    if combo in ("H", "I"):
        lines.append(f"Guest SMILES: {_fmt(row['guest_smiles'])}")
        lines.append(f"Guest tags: {_fmt(row['guest_tags'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Query 拼装
# ---------------------------------------------------------------------------


def _build_query(row: pd.Series, combo: str, reasoning: str) -> str:
    fields = _render_fields(row, combo)
    parts: list[str] = [GUIDANCE, "", fields]
    if reasoning == "cot_structured":
        parts += ["", STRUCTURED_REASONING]
    parts += ["", "Choose exactly ONE solvent class:", OPTIONS_BLOCK]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Few-shot 示例选择 — 复用 Task 2 的逻辑,但渲染用新字段
# ---------------------------------------------------------------------------


def pick_fewshot_examples(df: pd.DataFrame) -> pd.DataFrame:
    chosen: list[pd.Series] = []
    used_hosts: set = set()
    for lab in LABELS_ORDER:
        sub = df[df["solvent_label"] == lab].copy()
        if len(sub) == 0:
            continue
        # 优先选有 guest_smiles 的、host 未被占用的、id 靠前的
        sub["_has_gsmi"] = sub["guest_smiles"].apply(
            lambda v: 0 if (v is None or (isinstance(v, float) and pd.isna(v))) else 1
        )
        sub["_new_host"] = sub["host"].apply(lambda h: 0 if h in used_hosts else 1)
        sub = sub.sort_values(
            ["_has_gsmi", "_new_host", "interaction_id"],
            ascending=[False, False, True],
        )
        pick = sub.iloc[0]
        chosen.append(pick)
        used_hosts.add(pick["host"])
    return pd.DataFrame(chosen)


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


def render(
    row: pd.Series,
    combo: str,
    reasoning: str,
    shot: str,
    examples: pd.DataFrame | None,
) -> str:
    query = _build_query(row, combo, reasoning)
    thinking = reasoning in ("cot_vanilla", "cot_structured")
    fewshot_examples: list[dict[str, str]] | None = None
    if shot == "few" and examples is not None and len(examples) > 0:
        fewshot_examples = []
        for _, ex in examples.iterrows():
            ex_query = _build_query(ex, combo, reasoning)
            fewshot_examples.append(
                {"query": ex_query, "answer": LETTER_MAP[str(ex["solvent_label"])]}
            )
    return generate_prompt(query, fewshot_examples=fewshot_examples, thinking=thinking)


# ---------------------------------------------------------------------------
# 单文件构建
# ---------------------------------------------------------------------------


def _out_filename(combo: str, reasoning: str, shot: str, limit: int | None) -> str:
    base = f"task3_combo_{combo}_{reasoning}_{shot}"
    if limit:
        base += f"_first{limit}"
    return base + ".jsonl"


def build_one(
    df: pd.DataFrame,
    examples_full: pd.DataFrame,
    combo: str,
    reasoning: str,
    shot: str,
    out_dir: Path,
    limit: int | None,
) -> tuple[Path, int, list[int]]:
    examples = None
    df_eval = df
    if shot == "few":
        examples = examples_full
        excluded = set(examples["interaction_id"].astype(int).tolist())
        df_eval = df_eval[~df_eval["interaction_id"].isin(excluded)].copy()
    if limit is not None:
        df_eval = df_eval.head(limit).copy()

    out_path = out_dir / _out_filename(combo, reasoning, shot, limit)
    n = 0
    lengths: list[int] = []
    with out_path.open("w") as f:
        for _, row in df_eval.iterrows():
            p = render(row, combo, reasoning, shot, examples)
            true_label = str(row["solvent_label"])
            rec = {
                "interaction_id": int(row["interaction_id"]),
                "true_label": true_label,
                "true_letter": LETTER_MAP[true_label],
                "prompt": p,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            lengths.append(len(p))
            n += 1
    return out_path, n, lengths


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main(
    root: Path,
    combo: str,
    reasoning: str,
    shot: str,
    limit: int | None,
    do_all: bool,
) -> int:
    src = root / "data" / "task7" / "eval.parquet"
    out_dir = root / "prompts"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src)
    examples_full = pick_fewshot_examples(df)

    # 写出 few-shot 示例清单
    ex_meta = out_dir / "task3_fewshot_examples.json"
    ex_meta.write_text(
        json.dumps(
            [
                {
                    "interaction_id": int(r["interaction_id"]),
                    "label": str(r["solvent_label"]),
                    "letter": LETTER_MAP[str(r["solvent_label"])],
                    "host": str(r["host"]),
                    "guest": str(r["guest"]),
                }
                for _, r in examples_full.iterrows()
            ],
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"[fewshot] 示例清单 → {ex_meta} ({len(examples_full)} 条)")

    if do_all:
        total = 0
        for c in COMBOS:
            for r in REASONINGS:
                for s in SHOTS:
                    out_path, n, lengths = build_one(
                        df, examples_full, c, r, s, out_dir, limit
                    )
                    print(
                        f"  → {out_path.name}: {n} 行, "
                        f"prompt 字符 min={min(lengths)} med={sorted(lengths)[n//2]} max={max(lengths)}"
                    )
                    total += 1
        print(f"\n[all] 共生成 {total} 个 JSONL 到 {out_dir}/")
        return 0

    out_path, n, lengths = build_one(
        df, examples_full, combo, reasoning, shot, out_dir, limit
    )
    print(f"\n写入 {out_path} · {n} 行")
    print(
        f"prompt 字符数: min={min(lengths)}  median={sorted(lengths)[n//2]}  max={max(lengths)}"
    )

    # 抽检
    print("\n=== 抽检 (首 / 中 / 尾) ===")
    with out_path.open() as f:
        records = [json.loads(line) for line in f]
    for name, idx in (("FIRST", 0), ("MID", n // 2), ("LAST", n - 1)):
        rec = records[idx]
        print(
            f"\n--- {name} (interaction_id={rec['interaction_id']}, "
            f"true={rec['true_label']}={rec['true_letter']}) ---"
        )
        p = rec["prompt"]
        print(p[:1500] + ("\n... [truncated]" if len(p) > 1500 else ""))
        print("-" * 60)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--combo", default="G", choices=COMBOS)
    ap.add_argument(
        "--reasoning", default="none",
        choices=REASONINGS,
    )
    ap.add_argument("--shot", default="zero", choices=SHOTS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    sys.exit(
        main(
            Path(args.root), args.combo, args.reasoning, args.shot,
            args.limit, args.all,
        )
    )
