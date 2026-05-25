"""SID dataset loader — solvent compatibility classification (6-class MCQ).

Builds prompts in combo-H style: host SMILES + guest name + guest SMILES +
guest tags, wrapped with the canonical :func:`templates.generate_prompt`.

Config keys consumed
--------------------
data_path : str
    Path to ``data/sid/eval.parquet``.
prompt.fewshot_k : int (default 0)
    Number of few-shot examples to include. If > 0, rows are loaded from
    ``data/sid/fewshot_examples.json`` and matched against the parquet for
    their full fields.  Fewshot rows are excluded from the evaluation set.
prompt.thinking : bool (default False)
    If True, the chain-of-thought cue is appended to the prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from templates import generate_options, generate_prompt

from .base import Example, SupraDataset, register_dataset

# ---------------------------------------------------------------------------
# Label set — 6 canonical solvent classes, fixed order A-F
# ---------------------------------------------------------------------------

LABELS_ORDER = ["water", "DMSO", "MeCN", "MeOH", "CHCl3", "CH2Cl2"]
LETTER_MAP: dict[str, str] = {lab: chr(ord("A") + i) for i, lab in enumerate(LABELS_ORDER)}
OPTIONS_BLOCK = generate_options(LABELS_ORDER)

# ---------------------------------------------------------------------------
# Guidance text (combo-H / SMILES-based host context)
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


def _fmt(v: object) -> str:
    """Stringify a value, returning 'N/A' for None / NaN."""
    import math

    if v is None:
        return "N/A"
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    s = str(v).strip()
    return s if s else "N/A"


def _build_query(row: "pd.Series") -> str:  # type: ignore[name-defined]  # noqa: F821
    """Render the combo-H query for one row."""
    fields = (
        f"Host SMILES: {_fmt(row['host_smiles'])}\n"
        f"Guest name: {_fmt(row['guest'])}\n"
        f"Guest SMILES: {_fmt(row['guest_smiles'])}\n"
        f"Guest tags: {_fmt(row['guest_tags'])}"
    )
    parts = [GUIDANCE, "", fields, "", "Choose exactly ONE solvent class:", OPTIONS_BLOCK]
    return "\n".join(parts)


@register_dataset("sid")
class Task7Dataset(SupraDataset):
    """6-class solvent compatibility classification (MCQ, combo-H prompts)."""

    def __iter__(self) -> Iterator[Example]:
        import pandas as pd  # lazy import — pandas is heavy

        data_path = Path(self.config["data_path"])
        prompt_cfg = self.config.get("prompt", {}) or {}
        fewshot_k = int(prompt_cfg.get("fewshot_k", 0))
        thinking = bool(prompt_cfg.get("thinking", False))

        df = pd.read_parquet(data_path)

        # ------------------------------------------------------------------
        # Load few-shot examples (if requested)
        # ------------------------------------------------------------------
        fewshot_examples: list[dict[str, str]] | None = None
        excluded_ids: set[int] = set()

        if fewshot_k > 0:
            fewshot_meta_path = data_path.parent / "fewshot_examples.json"
            with fewshot_meta_path.open("r", encoding="utf-8") as fh:
                fewshot_meta = json.load(fh)

            # Slice to the requested k
            fewshot_meta = fewshot_meta[:fewshot_k]
            fs_ids = {int(ex["interaction_id"]) for ex in fewshot_meta}
            excluded_ids = fs_ids

            # Look up full rows from parquet for combo-H rendering
            fs_rows = df[df["interaction_id"].isin(fs_ids)].set_index("interaction_id")
            fewshot_examples = []
            for ex in fewshot_meta:
                iid = int(ex["interaction_id"])
                if iid not in fs_rows.index:
                    continue
                row = fs_rows.loc[iid]
                fewshot_examples.append(
                    {
                        "query": _build_query(row),
                        "answer": LETTER_MAP[str(ex["label"])],
                    }
                )

        # ------------------------------------------------------------------
        # Iterate eval rows, excluding few-shot IDs
        # ------------------------------------------------------------------
        idx = 0
        for _, row in df.iterrows():
            if int(row["interaction_id"]) in excluded_ids:
                continue
            if self.limit is not None and idx >= self.limit:
                break

            true_label = str(row["solvent_label"])
            true_letter = LETTER_MAP.get(true_label, "?")
            query = _build_query(row)
            prompt = generate_prompt(
                query,
                fewshot_examples=fewshot_examples,
                thinking=thinking,
            )

            yield Example(
                id=str(int(row["interaction_id"])),
                prompt=prompt,
                reference=true_label,
                metadata={
                    "true_letter": true_letter,
                    "host_family": str(row.get("host_family", "")),
                    "host_smiles": str(row["host_smiles"]),
                    "guest_name": str(row["guest"]),
                    "guest_smiles": _fmt(row["guest_smiles"]),
                    "guest_tags": _fmt(row["guest_tags"]),
                },
            )
            idx += 1
