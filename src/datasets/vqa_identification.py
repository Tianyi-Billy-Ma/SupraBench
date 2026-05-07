"""VQA identification dataset — predict canonical SMILES from a single
2D structural drawing of a molecule.

Reads ``identification.csv`` (one row per molecule) and yields one
:class:`Example` per row that has a non-empty ``cano_smiles`` reference.
Each example carries a single PIL image on ``Example.images``.

Three prompting modes (set ``prompt.mode`` in the task YAML; mutually
exclusive):

* ``base``  — :func:`templates.generate_prompt` only, no extras.
* ``fewshot`` — same rendered prompt as ``base`` plus ``fewshot_k``
  in-context demonstrations on ``Example.fewshot_demos``. Demos are
  sampled from the eval set itself with the current row's id excluded;
  sampling is deterministic given ``prompt.seed``.
* ``cot``  — :func:`templates.generate_prompt` with ``thinking=True``
  (appends "Let's think step by step.").

Config (from ``configs/tasks/vqa_identification.yaml``)::

    dataset: vqa_identification
    data_path: supra-vqa/identification.csv
    data_root: supra-vqa
    prompt:
      mode: base       # base | fewshot | cot
      fewshot_k: 3     # only used when mode == fewshot
      seed: 42         # fewshot sampling seed

CSV columns required: ``molecule_id``, ``image``, ``cano_smiles``.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path
from typing import Any, Iterator

from templates import generate_prompt

from .base import Example, SupraDataset, register_dataset


_QUERY = (
    "You are shown the 2D structural drawing of a single small molecule. "
    "Reply with ONLY the SMILES string for this molecule. "
    "Output one line containing only the SMILES. No explanation, no prefix, "
    "no backticks, no code fences."
)

_VALID_MODES = {"base", "fewshot", "cot"}


def _load_image(path: Path):
    """Open a PIL RGB image; return None on failure (caller logs/skips)."""
    from PIL import Image
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        print(f"[skip image] {path}: {exc}", file=sys.stderr)
        return None


@register_dataset("vqa_identification")
class VQAIdentificationDataset(SupraDataset):
    def __iter__(self) -> Iterator[Example]:
        data_path = Path(self.config["data_path"])
        data_root = Path(self.config.get("data_root", data_path.parent))
        prompt_cfg = self.config.get("prompt", {}) or {}

        mode = prompt_cfg.get("mode", "base")
        if mode not in _VALID_MODES:
            raise ValueError(
                f"invalid prompt.mode {mode!r}; must be one of {sorted(_VALID_MODES)}"
            )
        fewshot_k = int(prompt_cfg.get("fewshot_k", 3))
        seed = int(prompt_cfg.get("seed", 42))

        prompt = generate_prompt(_QUERY, thinking=(mode == "cot"))

        # --- pass 1: collect all rows whose metadata makes them eligible
        with data_path.open("r", encoding="utf-8") as fh:
            all_rows = list(csv.DictReader(fh))
        eligible: list[dict[str, Any]] = [
            r for r in all_rows if (r.get("cano_smiles") or "").strip()
        ]

        if mode == "fewshot" and len(eligible) < fewshot_k + 1:
            raise ValueError(
                f"vqa_identification fewshot mode needs at least k+1 = {fewshot_k + 1} "
                f"eligible rows; only {len(eligible)} available"
            )

        # --- pass 2: yield, honouring self.limit
        yielded = 0
        for row in eligible:
            if self.limit is not None and yielded >= self.limit:
                break

            row_id = row["molecule_id"]
            query_img = _load_image(data_root / row["image"])
            if query_img is None:
                continue

            fewshot_demos: list[dict[str, Any]] | None = None
            if mode == "fewshot":
                fewshot_demos = self._sample_demos(
                    eligible, exclude_id=row_id, k=fewshot_k,
                    seed=seed, data_root=data_root,
                )
                if fewshot_demos is None:
                    print(f"[skip] {row_id}: could not assemble {fewshot_k} demos",
                          file=sys.stderr)
                    continue

            yield Example(
                id=row_id,
                prompt=prompt,
                reference=row["cano_smiles"].strip(),
                images=[query_img],
                metadata={"image": row["image"]},
                fewshot_demos=fewshot_demos,
            )
            yielded += 1

    @staticmethod
    def _sample_demos(
        pool: list[dict[str, Any]],
        exclude_id: str,
        k: int,
        seed: int,
        data_root: Path,
    ) -> list[dict[str, Any]] | None:
        """Deterministically pick k demos that load OK, excluding ``exclude_id``.

        Returns None if the pool is exhausted before k usable demos are found.
        """
        rng = random.Random(hash((seed, exclude_id)))
        candidates = [r for r in pool if r["molecule_id"] != exclude_id]
        rng.shuffle(candidates)

        demos: list[dict[str, Any]] = []
        for cand in candidates:
            if len(demos) == k:
                break
            img = _load_image(data_root / cand["image"])
            if img is None:
                continue
            demos.append({
                "answer": cand["cano_smiles"].strip(),
                "images": [img],
            })
        return demos if len(demos) == k else None
