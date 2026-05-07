"""Task 3 dataset loader — host/guest property explanation (open QA).

Loads ``data/task3/task3.jsonl`` (134 rows, subtypes: forward / reverse),
splits off a per-subtype few-shot pool (6 rows each, lowest ids), and
renders every test row through :func:`templates.generate_prompt` so the
prompt layout is identical to every other SupraBench task.

Schema of each source row::

    {id, task, subtype, question, answer,
     host_name, guest_name?, guest_smiles?,
     n_guests_smi?, n_top?, gt_mw_mean?, gt_mw_std?,
     gt_charge?, gt_rings_mean?,
     n_hosts?, n_top_hosts?, max_logka?}

The ``question`` field already contains a fully-worded prompt wrapped in the
standard preamble.  We strip the preamble and the trailing answer-tag
instruction to recover the bare question, then re-render it through the
shared template.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from templates import generate_prompt

from .base import Example, SupraDataset, register_dataset

K_FEWSHOT = 6
SUBTYPES = ("forward", "reverse")

_QUESTION_PREFIX = (
    "You are an expert in supramolecular chemistry.\n"
    "Your task is to answer the following question.\n"
    "Question: "
)
_ANSWER_SUFFIX = "Put your final answer between <answer></answer>"


def _extract_inner_question(full_question: str) -> str:
    """Strip preamble and answer-tag instruction from a source question field."""
    s = full_question
    if s.startswith(_QUESTION_PREFIX):
        s = s[len(_QUESTION_PREFIX):]
    idx = s.rfind(_ANSWER_SUFFIX)
    if idx != -1:
        s = s[:idx]
    return s.strip()


def _build_query(row: dict) -> str:
    """Compose the user-facing query string from a source row.

    Appends host/guest names and SMILES when available so the prompt is
    self-contained even without few-shot examples.
    """
    inner_q = _extract_inner_question(row["question"])
    parts = [inner_q]

    host_name = row.get("host_name")
    guest_name = row.get("guest_name")
    guest_smiles = row.get("guest_smiles")

    if host_name and host_name not in inner_q:
        parts.append(f"Host: {host_name}")
    if guest_name and guest_name not in inner_q:
        parts.append(f"Guest: {guest_name}")
    if guest_smiles and guest_smiles not in inner_q:
        parts.append(f"Guest SMILES: {guest_smiles}")

    return "\n".join(parts)


@register_dataset("task3")
class Task3Dataset(SupraDataset):
    def __iter__(self) -> Iterator[Example]:
        path = Path(self.config["data_path"])
        prompt_cfg = self.config.get("prompt", {}) or {}
        fewshot_k = int(prompt_cfg.get("fewshot_k", 0))
        thinking = bool(prompt_cfg.get("thinking", False))

        with path.open("r", encoding="utf-8") as fh:
            all_rows = [json.loads(line) for line in fh if line.strip()]

        # Partition into per-subtype few-shot pool and test set.
        by_subtype: dict[str, list[dict]] = {st: [] for st in SUBTYPES}
        for row in all_rows:
            st = row.get("subtype", "")
            if st in by_subtype:
                by_subtype[st].append(row)

        fewshot_pool: dict[str, list[dict]] = {}
        fewshot_ids: set[str] = set()
        for st in SUBTYPES:
            sorted_rows = sorted(by_subtype[st], key=lambda r: r["id"])
            fewshot_pool[st] = sorted_rows[:K_FEWSHOT]
            fewshot_ids.update(r["id"] for r in fewshot_pool[st])

        test_rows = [r for r in all_rows if r["id"] not in fewshot_ids]

        yielded = 0
        for row in test_rows:
            if self.limit is not None and yielded >= self.limit:
                break

            subtype = row.get("subtype", "")
            query = _build_query(row)

            # Build few-shot examples for this row's subtype.
            fewshot_examples = None
            if fewshot_k > 0 and subtype in fewshot_pool:
                pool = fewshot_pool[subtype][:fewshot_k]
                fewshot_examples = [
                    {
                        "query": _extract_inner_question(ex["question"]),
                        "answer": ex["answer"],
                    }
                    for ex in pool
                ]

            prompt = generate_prompt(
                query,
                fewshot_examples=fewshot_examples,
                thinking=thinking,
            )

            yield Example(
                id=row["id"],
                prompt=prompt,
                reference=row["answer"],
                metadata={
                    "subtype": subtype,
                    "host_name": row.get("host_name"),
                    "guest_name": row.get("guest_name"),
                    "guest_smiles": row.get("guest_smiles"),
                    "gt_mw_mean": row.get("gt_mw_mean"),
                    "gt_mw_std": row.get("gt_mw_std"),
                    "gt_charge": row.get("gt_charge"),
                    "gt_rings_mean": row.get("gt_rings_mean"),
                    "n_guests_smi": row.get("n_guests_smi"),
                    "n_top": row.get("n_top"),
                    "n_hosts": row.get("n_hosts"),
                    "n_top_hosts": row.get("n_top_hosts"),
                    "max_logka": row.get("max_logka"),
                },
            )
            yielded += 1
