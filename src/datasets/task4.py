"""Task 4 dataset loader — logKa VQA (host + guest images → float logKa).

Reads a CSV with one row per host-guest pair. Expected columns::

    pair_id, host_id, guest_id, host_image, guest_image, gold_logka

``host_image`` and ``guest_image`` are paths relative to ``image_root``
(both configured in ``configs/tasks/task4.yaml``).

**Data layout** — before running, populate::

    data/task4/logka.csv          # the eval CSV described above
    data/task4/images/            # image files referenced in the CSV

If ``data_path`` does not exist, a descriptive error is raised pointing the
user to the correct location.

**Image inputs** — ``metadata["images"]`` carries ``[host_image_path,
guest_image_path]`` as ``str`` values for each example. The current
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

# Prompt shown to the model (images are passed separately by a multimodal
# backend; this text-only form is what the current backends receive).
_LOGKA_PROMPT = (
    "The first image is a HOST molecule. The second image is its GUEST molecule. "
    "Estimate logKa = log10(Ka) for the 1:1 host-guest complex in water at ~25 C, "
    "where Ka is the association constant in M^-1. "
    "Reply with ONLY the value of logKa as a single decimal number, "
    "typically in the range -3 to 16. "
    "Do NOT output the raw Ka, do NOT use scientific notation, "
    "no units, no explanation, no prefix."
)


@register_dataset("task4")
class Task4Dataset(SupraDataset):
    """logKa VQA dataset: host + guest molecular images → logKa float.

    Config fields consumed from the task YAML:
    - ``data_path``: path to the eval CSV (absolute or relative to CWD).
    - ``image_root``: directory containing the image files referenced in the CSV.
    - ``prompt.thinking``: bool, passed through to :func:`templates.generate_prompt`.
    """

    def __iter__(self) -> Iterator[Example]:
        data_path = Path(self.config["data_path"])
        if not data_path.exists():
            raise FileNotFoundError(
                f"Task 4 data file not found: {data_path}\n"
                "Please populate data/task4/logka.csv with columns:\n"
                "  pair_id, host_id, guest_id, host_image, guest_image, gold_logka\n"
                "and place corresponding images under data/task4/images/."
            )

        image_root = Path(self.config.get("image_root", "data/task4/images"))
        prompt_cfg = self.config.get("prompt", {}) or {}
        thinking = bool(prompt_cfg.get("thinking", False))

        prompt = generate_prompt(_LOGKA_PROMPT, thinking=thinking)

        with data_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for idx, row in enumerate(reader):
                if self.limit is not None and idx >= self.limit:
                    break
                host_img = str(image_root / row["host_image"])
                guest_img = str(image_root / row["guest_image"])
                yield Example(
                    id=row["pair_id"],
                    prompt=prompt,
                    reference=float(row["gold_logka"]),
                    metadata={
                        "host_id": row.get("host_id", ""),
                        "guest_id": row.get("guest_id", ""),
                        "images": [host_img, guest_img],
                    },
                )
