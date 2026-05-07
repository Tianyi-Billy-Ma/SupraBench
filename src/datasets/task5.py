"""Task 5 dataset loader — molecular-structure identification VQA.

Given a 2D structural image of a single molecule, the model must predict
either its **name** or its **canonical SMILES** string, depending on the
``mode`` field in the task YAML.

Reads a CSV with one row per molecule. Expected columns::

    molecule_id, image, gold_name, names_alias_set, gold_smiles

- ``image`` is a path relative to ``image_root``.
- ``names_alias_set`` is a pipe-separated (``|``) list of accepted name
  aliases used during evaluation (name mode only).
- ``gold_smiles`` may be empty for molecules where no canonical SMILES is
  available; those rows are included in name-mode but skipped in smiles-mode.

**Data layout** — before running, populate::

    data/task5/identification.csv   # the eval CSV described above
    data/task5/images/              # image files referenced in the CSV

If ``data_path`` does not exist, a descriptive error is raised pointing the
user to the correct location.

**Image inputs** — ``metadata["images"]`` carries ``[molecule_image_path]``
as a single-element list of ``str``. The current
:class:`~inference.base.InferenceBackend` only accepts text via
``generate(prompt: str)``. Image inputs in ``metadata.images`` are **not
yet honored** by any inference backend; backends need a follow-up extension
to read ``metadata.images`` and pass them to the multimodal processor.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from templates import generate_prompt

from .base import Example, SupraDataset, register_dataset

# Prompts mirroring the PROMPTS dict from the original run_identification.py.
_PROMPTS: dict[str, str] = {
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

_VALID_MODES = frozenset(_PROMPTS)


@register_dataset("task5")
class Task5Dataset(SupraDataset):
    """Molecular-structure identification VQA dataset.

    Config fields consumed from the task YAML:
    - ``data_path``: path to the eval CSV (absolute or relative to CWD).
    - ``image_root``: directory containing the image files referenced in the CSV.
    - ``mode``: ``"name"`` (predict molecule name) or ``"smiles"`` (predict SMILES).
    - ``prompt.thinking``: bool, passed through to :func:`templates.generate_prompt`.

    In ``smiles`` mode, rows with an empty ``gold_smiles`` field are skipped.
    """

    def __iter__(self) -> Iterator[Example]:
        data_path = Path(self.config["data_path"])
        if not data_path.exists():
            raise FileNotFoundError(
                f"Task 5 data file not found: {data_path}\n"
                "Please populate data/task5/identification.csv with columns:\n"
                "  molecule_id, image, gold_name, names_alias_set, gold_smiles\n"
                "and place corresponding images under data/task5/images/."
            )

        mode = self.config.get("mode", "smiles")
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Task 5 'mode' must be one of {sorted(_VALID_MODES)}; got {mode!r}"
            )

        image_root = Path(self.config.get("image_root", "data/task5/images"))
        prompt_cfg = self.config.get("prompt", {}) or {}
        thinking = bool(prompt_cfg.get("thinking", False))

        prompt = generate_prompt(_PROMPTS[mode], thinking=thinking)

        with data_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            emitted = 0
            for row in reader:
                if self.limit is not None and emitted >= self.limit:
                    break

                # In smiles mode, skip rows without a gold SMILES.
                if mode == "smiles" and not row.get("gold_smiles", "").strip():
                    continue

                mol_img = str(image_root / row["image"])
                aliases = [
                    a.strip()
                    for a in row.get("names_alias_set", "").split("|")
                    if a.strip()
                ]

                if mode == "name":
                    reference: str = row["gold_name"]
                    meta: dict = {
                        "images": [mol_img],
                        "aliases": aliases,
                    }
                else:
                    reference = row["gold_smiles"]
                    meta = {"images": [mol_img]}

                yield Example(
                    id=row["molecule_id"],
                    prompt=prompt,
                    reference=reference,
                    metadata=meta,
                )
                emitted += 1
