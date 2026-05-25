"""Build the v2 CPT mix dataset and push it to ``mtybilly/SupraBench-CPT-Mix-v2``.

Recipe (informed by Qi et al. 2025, EvoLM, arxiv 2506.16029):

    | Stream         | Source                              | Target % of tokens |
    | -------------- | ----------------------------------- | ------------------ |
    | domain         | mtybilly/EU-PMC[supramolecular]     | 80                 |
    | replay         | HuggingFaceFW/fineweb-edu (sample-10BT) | 15             |
    | format-anchor  | allenai/tulu-3-sft-mixture          | 5                  |

EvoLM's Table 2 shows that ~5–16 % replay is the sweet spot at 50 BT scale
on a 1B base model. We sit inside that band (15 % combined replay+anchor),
with one deliberate addition: a small format-anchor stream of instruction-
formatted text concatenated as raw `prompt\\n\\nresponse` pairs (no chat
template). EvoLM uses raw text only; we add ~5 % anchor specifically to
counter the instruction-following collapse we observed in v1
(SID mode-collapsed onto majority class, TBS dropped 14 pts).

Outputs a single split ``train`` with columns ``text, source, length`` so
the trainer can iterate naively while keeping per-stream provenance.
"""

from __future__ import annotations

import argparse
import random
from typing import Iterator

from datasets import (
    Dataset,
    Features,
    Value,
    load_dataset,
)
from huggingface_hub import whoami
from transformers import AutoTokenizer

DOMAIN_REPO = "mtybilly/EU-PMC"
DOMAIN_CONFIG = "supramolecular"
DOMAIN_SPLIT = "train"

REPLAY_REPO = "HuggingFaceFW/fineweb-edu"
REPLAY_CONFIG = "sample-10BT"
REPLAY_SPLIT = "train"

ANCHOR_REPO = "allenai/tulu-3-sft-mixture"
ANCHOR_SPLIT = "train"

TARGET_REPO = "mtybilly/SupraBench-CPT-Mix-v2"

DEFAULT_RATIOS = {"domain": 0.80, "replay": 0.15, "format_anchor": 0.05}


def _tokenizer(model_id: str = "Qwen/Qwen3.5-27B"):
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def _domain_iter(tok) -> Iterator[tuple[str, int]]:
    ds = load_dataset(DOMAIN_REPO, name=DOMAIN_CONFIG, split=DOMAIN_SPLIT)
    for row in ds:
        title = row.get("title") or ""
        abstract = row.get("abstract") or ""
        text = f"{title}\n\n{abstract}".strip()
        if not text:
            continue
        n = len(tok(text, add_special_tokens=False).input_ids)
        yield text, n


def _replay_iter(tok, target_tokens: int, seed: int) -> Iterator[tuple[str, int]]:
    """Sample FineWeb-Edu in streaming mode, yielding until we hit target_tokens."""
    ds = load_dataset(REPLAY_REPO, name=REPLAY_CONFIG, split=REPLAY_SPLIT, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    seen = 0
    for row in ds:
        text = (row.get("text") or "").strip()
        if len(text) < 200:
            continue
        n = len(tok(text, add_special_tokens=False).input_ids)
        seen += n
        yield text, n
        if seen >= target_tokens:
            break


def _anchor_iter(tok, target_tokens: int, seed: int) -> Iterator[tuple[str, int]]:
    """Sample Tulu-3-SFT-mixture; flatten messages into raw `prompt\\n\\nresponse`."""
    ds = load_dataset(ANCHOR_REPO, split=ANCHOR_SPLIT, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    seen = 0
    for row in ds:
        msgs = row.get("messages") or []
        parts = []
        for m in msgs:
            role = m.get("role", "")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            tag = {"user": "Question", "assistant": "Answer", "system": "Context"}.get(role, role.title())
            parts.append(f"{tag}: {content}")
        if not parts:
            continue
        text = "\n\n".join(parts)
        n = len(tok(text, add_special_tokens=False).input_ids)
        seen += n
        yield text, n
        if seen >= target_tokens:
            break


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ratios", type=str, default="0.80,0.15,0.05",
                   help="domain,replay,format_anchor token-fraction targets")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Build in-memory, print stats, do not push.")
    p.add_argument("--commit-message", type=str,
                   default="v2 CPT mix: 80%% supra-EU-PMC + 15%% FineWeb-Edu + 5%% Tulu-3 anchor (EvoLM recipe)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    user = whoami().get("name")
    print(f"Authenticated as: {user}")
    if not args.dry_run and user != "mtybilly":
        raise SystemExit("Must be authenticated as mtybilly to push to mtybilly/SupraBench-CPT-Mix-v2.")

    rd, rr, rf = (float(x) for x in args.ratios.split(","))
    if abs((rd + rr + rf) - 1.0) > 1e-6:
        raise SystemExit(f"Ratios must sum to 1.0, got {rd + rr + rf}")
    print(f"Target ratios: domain={rd:.2f} replay={rr:.2f} format_anchor={rf:.2f}")

    tok = _tokenizer()
    print(f"Tokenizer: {tok.__class__.__name__} | vocab={tok.vocab_size}")

    # Pass 1: pull all of the domain stream first; that fixes the absolute
    # token budget. The other streams are sized as a fraction of total.
    domain_records: list[dict] = []
    domain_tokens = 0
    print("\n[domain] iterating supramolecular subset ...")
    for text, n in _domain_iter(tok):
        domain_records.append({"text": text, "source": "domain", "length": n})
        domain_tokens += n
    print(f"[domain] rows={len(domain_records):,}  tokens={domain_tokens:,}")

    total_tokens = int(domain_tokens / rd)
    target_replay = int(total_tokens * rr)
    target_anchor = int(total_tokens * rf)
    print(f"\nImplied total = {total_tokens:,} tokens")
    print(f"  target replay = {target_replay:,}")
    print(f"  target anchor = {target_anchor:,}")

    print("\n[replay] streaming FineWeb-Edu ...")
    replay_records: list[dict] = []
    replay_tokens = 0
    for text, n in _replay_iter(tok, target_replay, seed=args.seed):
        replay_records.append({"text": text, "source": "replay", "length": n})
        replay_tokens += n
    print(f"[replay] rows={len(replay_records):,}  tokens={replay_tokens:,}")

    print("\n[anchor] streaming Tulu-3-SFT-mixture ...")
    anchor_records: list[dict] = []
    anchor_tokens = 0
    for text, n in _anchor_iter(tok, target_anchor, seed=args.seed + 1):
        anchor_records.append({"text": text, "source": "format_anchor", "length": n})
        anchor_tokens += n
    print(f"[anchor] rows={len(anchor_records):,}  tokens={anchor_tokens:,}")

    rng = random.Random(args.seed)
    all_records = domain_records + replay_records + anchor_records
    rng.shuffle(all_records)
    grand_total = domain_tokens + replay_tokens + anchor_tokens
    print(f"\nFinal mix: rows={len(all_records):,}  tokens={grand_total:,}")
    print(f"  domain = {100 * domain_tokens / grand_total:.1f}%")
    print(f"  replay = {100 * replay_tokens / grand_total:.1f}%")
    print(f"  anchor = {100 * anchor_tokens / grand_total:.1f}%")

    features = Features({
        "text": Value("string"),
        "source": Value("string"),
        "length": Value("int32"),
    })
    ds = Dataset.from_list(all_records, features=features)

    if args.dry_run:
        print("\n(--dry-run set, not pushing)")
        return

    print(f"\nPushing to {TARGET_REPO} ...")
    ds.push_to_hub(
        repo_id=TARGET_REPO,
        commit_message=args.commit_message,
        private=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
