"""
SID · Step 02 — Prompt rendering (host SMILES-only solvent prediction).

Same task as TBS, but the host is given as SMILES only (no name/family) to
test whether the LLM can infer the solvent purely from molecular structure.

Combo axis (input field stacking):
  G: host SMILES + guest name                       (minimal baseline)
  H: host SMILES + guest name + guest SMILES + tags (guest structure given)
  I: host SMILES + guest SMILES + guest tags        (pure structure, no names)

Other axes (same as TBS):
  - reasoning: none / cot_vanilla / cot_structured
  - shot:      zero / few

Output JSONL per line: {interaction_id, true_label, true_letter, prompt}

Usage:
    python scripts/sid_data_prep/02_build_prompts_sid.py --combo G --reasoning none --shot zero
    python scripts/sid_data_prep/02_build_prompts_sid.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "SupraBench" / "src"))
from templates import generate_options, generate_prompt  # noqa: E402


LABELS_ORDER = ["water", "DMSO", "MeCN", "MeOH", "CHCl3", "CH2Cl2"]
LETTER_MAP: dict[str, str] = {lab: chr(ord("A") + i) for i, lab in enumerate(LABELS_ORDER)}
OPTIONS_BLOCK = generate_options(LABELS_ORDER)

COMBOS = ["G", "H", "I"]
REASONINGS = ["none", "cot_vanilla", "cot_structured"]
SHOTS = ["zero", "few"]


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


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float) and pd.isna(v):
        return "N/A"
    s = str(v).strip()
    return s if s else "N/A"


def _render_fields(row: pd.Series, combo: str) -> str:
    lines = [f"Host SMILES: {_fmt(row['host_smiles'])}"]
    if combo in ("G", "H"):
        lines.append(f"Guest name: {_fmt(row['guest'])}")
    if combo in ("H", "I"):
        lines.append(f"Guest SMILES: {_fmt(row['guest_smiles'])}")
        lines.append(f"Guest tags: {_fmt(row['guest_tags'])}")
    return "\n".join(lines)


def _build_query(row: pd.Series, combo: str, reasoning: str) -> str:
    fields = _render_fields(row, combo)
    parts: list[str] = [GUIDANCE, "", fields]
    if reasoning == "cot_structured":
        parts += ["", STRUCTURED_REASONING]
    parts += ["", "Choose exactly ONE solvent class:", OPTIONS_BLOCK]
    return "\n".join(parts)


def pick_fewshot_examples(df: pd.DataFrame) -> pd.DataFrame:
    """One example per label; prefer rows with guest SMILES and an unused host."""
    chosen: list[pd.Series] = []
    used_hosts: set = set()
    for lab in LABELS_ORDER:
        sub = df[df["solvent_label"] == lab].copy()
        if len(sub) == 0:
            continue
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


def _out_filename(combo: str, reasoning: str, shot: str, limit: int | None) -> str:
    base = f"sid_combo_{combo}_{reasoning}_{shot}"
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
        # Exclude few-shot examples from the eval set to prevent leakage.
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


def main(
    root: Path,
    combo: str,
    reasoning: str,
    shot: str,
    limit: int | None,
    do_all: bool,
) -> int:
    src = root / "data" / "sid" / "eval.parquet"
    out_dir = root / "prompts"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src)
    examples_full = pick_fewshot_examples(df)

    ex_meta = out_dir / "sid_fewshot_examples.json"
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
    print(f"[fewshot] examples -> {ex_meta} ({len(examples_full)} rows)")

    if do_all:
        total = 0
        for c in COMBOS:
            for r in REASONINGS:
                for s in SHOTS:
                    out_path, n, lengths = build_one(
                        df, examples_full, c, r, s, out_dir, limit
                    )
                    print(
                        f"  -> {out_path.name}: {n} rows, "
                        f"prompt chars min={min(lengths)} med={sorted(lengths)[n//2]} max={max(lengths)}"
                    )
                    total += 1
        print(f"\n[all] generated {total} JSONL files to {out_dir}/")
        return 0

    out_path, n, lengths = build_one(
        df, examples_full, combo, reasoning, shot, out_dir, limit
    )
    print(f"\nwrote {out_path} · {n} rows")
    print(
        f"prompt chars: min={min(lengths)}  median={sorted(lengths)[n//2]}  max={max(lengths)}"
    )

    print("\n=== sample (first / mid / last) ===")
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
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent.parent))
    ap.add_argument("--combo", default="G", choices=COMBOS)
    ap.add_argument("--reasoning", default="none", choices=REASONINGS)
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
