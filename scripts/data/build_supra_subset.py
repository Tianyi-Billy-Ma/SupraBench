"""Filter EU-PMC[filtered] down to a strict supramolecular-chemistry subset
and push it back as ``mtybilly/EU-PMC`` with config ``supramolecular``.

Why a stricter filter:
    The existing ``filtered`` split has 134K chemistry-leaning articles, but
    most are unrelated to host-guest / supramolecular work. v1 CPT (LoRA on
    Qwen3.5-27B + EU-PMC) showed catastrophic forgetting on SupraBench tasks,
    consistent with a corpus that was too broad. EvoLM (Qi et al. 2025,
    arxiv 2506.16029) shows that a denser, focused domain corpus beats a
    diluted larger one at fixed compute. We rebuild a tight subset whose
    title or abstract matches at least one supramolecular host or concept.

Usage::

    uv run --extra hf python scripts/data/build_supra_subset.py
    uv run --extra hf python scripts/data/build_supra_subset.py --dry-run
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

from datasets import load_dataset
from huggingface_hub import whoami

REPO_ID = "mtybilly/EU-PMC"
SOURCE_SPLIT = "filtered"
TARGET_CONFIG = "supramolecular"
TARGET_SPLIT = "train"

# Keyword groups. We keep the union (single hit suffices) — supramolecular
# papers usually mention at least one named host OR one concept term.
HOST_PATTERNS = [
    r"cucurbit\s*[\[\(]\s*\d+\s*[\]\)]\s*uril",
    r"cucurbituril",
    r"\bCB\s*[\[\(]\s*\d+\s*[\]\)]",
    r"cyclodextrin",
    r"\b[αβγ]-?cyclodextrin",
    r"\b[αβγ]-?CD\b",
    r"calix\s*[\[\(]\s*\d+\s*[\]\)]\s*arene",
    r"calixarene",
    r"pillar\s*[\[\(]\s*\d+\s*[\]\)]\s*arene",
    r"pillararene",
    r"\bcryptand\b",
    r"cryptophane",
    r"crown[- ]?ether",
    r"\bcrown-?\d+",
    r"cavitand",
    r"octa[- ]?acid",
    r"naphthotube",
    r"bambusuril",
    r"bambus\s*[\[\(]\s*\d+",
    r"glycoluril",
    r"pseudorotaxane",
    r"\brotaxane",
    r"\bcatenane",
    r"oxatube",
    r"resorcin\s*[\[\(]\s*\d+\s*[\]\)]\s*arene",
    r"resorcinarene",
]

CONCEPT_PATTERNS = [
    # Binding-specific concept terms only. Bare "supramolecular" /
    # "self-assembly" / "non-covalent" are intentionally excluded — they
    # catch large amounts of polymer, MOF, and coordination-chemistry
    # literature that has no host-guest content. The keyword set is
    # essentially "named host *or* binding-thermodynamics term".
    r"host[- ]?guest",
    r"inclusion[- ]?complex",
    r"molecular[- ]?recognition",
    r"binding[- ]?affinity",
    r"association[- ]?constant",
    r"binding[- ]?constant",
    r"isothermal[- ]?titration[- ]?calorimetry",
    r"\bITC\b",
    r"mechanically[- ]?interlock",
]

ALL_PATTERNS = HOST_PATTERNS + CONCEPT_PATTERNS
COMPILED = re.compile("|".join(f"(?:{p})" for p in ALL_PATTERNS), re.IGNORECASE)


def text_of(row: dict) -> str:
    title = row.get("title") or ""
    abstract = row.get("abstract") or ""
    return f"{title}\n{abstract}"


def keep(row: dict) -> bool:
    return bool(COMPILED.search(text_of(row)))


def first_hit(row: dict) -> str | None:
    m = COMPILED.search(text_of(row))
    return m.group(0).lower() if m else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Apply filter, print stats, do not push.")
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--commit-message", type=str,
                   default="Add supramolecular subset (strict keyword filter; EvoLM-aligned)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    user = whoami().get("name")
    print(f"Authenticated as: {user}")
    if user != "mtybilly":
        raise SystemExit("Must be authenticated as mtybilly to push to mtybilly/EU-PMC.")

    print(f"Loading {REPO_ID} split={SOURCE_SPLIT} ...")
    src = load_dataset(REPO_ID, split=SOURCE_SPLIT)
    print(f"  rows: {len(src):,}")
    print(f"  features: {src.features}")

    print("Filtering with strict supramolecular keyword regex ...")
    kept = src.filter(keep, num_proc=args.num_proc, desc="filter:supra")

    print(f"  kept: {len(kept):,}  ({100 * len(kept) / len(src):.1f}% of source)")

    sample_hits = Counter()
    for row in kept.select(range(min(len(kept), 5000))):
        h = first_hit(row)
        if h:
            sample_hits[h] += 1
    print("\nTop-30 first-keyword hits in first 5K kept rows:")
    for kw, n in sample_hits.most_common(30):
        print(f"  {n:5d}  {kw}")

    char_total = sum(len(text_of(r)) for r in kept)
    approx_tokens = char_total // 4
    print(f"\nApprox token count (char/4): {approx_tokens:,}")

    if args.dry_run:
        print("\n(--dry-run set, not pushing)")
        return

    print(f"\nPushing to {REPO_ID} config={TARGET_CONFIG} split={TARGET_SPLIT} ...")
    kept.push_to_hub(
        repo_id=REPO_ID,
        config_name=TARGET_CONFIG,
        split=TARGET_SPLIT,
        commit_message=args.commit_message,
        private=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
