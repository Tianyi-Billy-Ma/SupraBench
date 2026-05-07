"""VQA logKa dataset — predict log10(Ka) for a 1:1 host-guest complex.

Each row has a host molecule image and a guest molecule image, and a
gold ``logka_standard`` float reference. Yields one :class:`Example` per
row with two PIL images (host first, guest second) on
``Example.images``.

Three prompting modes (set ``prompt.mode`` in the task YAML; mutually
exclusive):

* ``base``  — :func:`templates.generate_prompt` only, no extras.
* ``fewshot`` — same rendered prompt as ``base`` plus ``fewshot_k``
  in-context demonstrations on ``Example.fewshot_demos``. Each demo
  carries its own ``[host_img, guest_img]`` pair plus the gold logKa as
  the answer text. Demos are sampled from the eval set itself with the
  current row's pair_id excluded; sampling is deterministic given
  ``prompt.seed``.
* ``cot``  — :func:`templates.generate_prompt` with ``thinking=True``
  (appends "Let's think step by step.").

Config (from ``configs/tasks/vqa_logka.yaml``)::

    dataset: vqa_logka
    data_path: supra-vqa/logka.csv
    data_root: supra-vqa
    prompt:
      mode: base       # base | fewshot | cot
      fewshot_k: 3     # only used when mode == fewshot
      seed: 42         # fewshot sampling seed

CSV columns required: ``pair_id``, ``host_image``, ``guest_image``,
``logka_standard``. ``host_id``, ``guest_id``, ``host_name``,
``guest_name`` are stored on metadata if present.
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
    "The first image is a HOST molecule. The second image is its GUEST molecule. "
    "Estimate logKa = log10(Ka) for the 1:1 host-guest complex in water at ~25 C, "
    "where Ka is the association constant in M^-1. "
    "Reply with ONLY the value of logKa as a single decimal number, "
    "typically in the range -3 to 16. "
    "Do NOT output the raw Ka, do NOT use scientific notation, "
    "no units, no explanation, no prefix."
)

_VALID_MODES = {"base", "fewshot", "cot"}


def _load_image(path: Path):
    from PIL import Image
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        print(f"[skip image] {path}: {exc}", file=sys.stderr)
        return None


def _parse_logka(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


@register_dataset("vqa_logka")
class VQALogKaDataset(SupraDataset):
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

        with data_path.open("r", encoding="utf-8") as fh:
            all_rows = list(csv.DictReader(fh))
        eligible: list[dict[str, Any]] = []
        for r in all_rows:
            gold = _parse_logka(r.get("logka_standard"))
            if gold is None:
                continue
            r = dict(r)
            r["_gold"] = gold
            eligible.append(r)

        if mode == "fewshot" and len(eligible) < fewshot_k + 1:
            raise ValueError(
                f"vqa_logka fewshot mode needs at least k+1 = {fewshot_k + 1} "
                f"eligible rows; only {len(eligible)} available"
            )

        yielded = 0
        for row in eligible:
            if self.limit is not None and yielded >= self.limit:
                break

            row_id = row["pair_id"]
            host_img = _load_image(data_root / row["host_image"])
            guest_img = _load_image(data_root / row["guest_image"])
            if host_img is None or guest_img is None:
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

            metadata = {
                "host_image": row["host_image"],
                "guest_image": row["guest_image"],
            }
            for k in ("host_id", "guest_id", "host_name", "guest_name", "source"):
                if row.get(k):
                    metadata[k] = row[k]

            yield Example(
                id=row_id,
                prompt=prompt,
                reference=row["_gold"],
                images=[host_img, guest_img],
                metadata=metadata,
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
        rng = random.Random(hash((seed, exclude_id)))
        candidates = [r for r in pool if r["pair_id"] != exclude_id]
        rng.shuffle(candidates)

        demos: list[dict[str, Any]] = []
        for cand in candidates:
            if len(demos) == k:
                break
            host = _load_image(data_root / cand["host_image"])
            guest = _load_image(data_root / cand["guest_image"])
            if host is None or guest is None:
                continue
            # Format gold to a clean decimal, matching what we ask the model to output.
            demos.append({
                "answer": f"{cand['_gold']:.4f}".rstrip("0").rstrip("."),
                "images": [host, guest],
            })
        return demos if len(demos) == k else None
