#!/usr/bin/env python3
"""Corpus quality validation via LLM-as-judge random sampling.

Samples 1000 articles each from the `raw` and `filtered` splits of the
mtybilly/EU-PMC Hugging Face dataset (default subset), asks Claude Haiku
4.5 (via OpenRouter) whether each article is centrally about supramolecular
chemistry, and reports per-split YES/NO/BORDERLINE rates plus a noise-floor
check on 50 of the samples re-judged a second time.

Usage:
    OPENROUTER_API_KEY=... uv run python tools/validate_corpus.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Literal

import httpx
from datasets import load_dataset

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-haiku-4.5"
DATASET_NAME = "mtybilly/EU-PMC"
DATASET_SUBSET = "default"
SAMPLES_PER_SPLIT = 1000
NOISE_FLOOR_N = 50
SEED = 42
MAX_CONCURRENCY = 16
OUT_DIR = Path("results/corpus_validation")

JUDGE_SYSTEM = (
    "You are a chemistry triage expert. Your job is to decide whether a "
    "given paper is centrally about supramolecular chemistry."
)

JUDGE_USER_TEMPLATE = """Given the paper's title and abstract below, decide whether the paper is centrally about *supramolecular chemistry* — the study of non-covalent host-guest associations between molecules, molecular recognition by macrocyclic hosts such as cucurbiturils, cyclodextrins, calixarenes, pillararenes, cryptands, or crown ethers, non-covalent inclusion complexes, or supramolecular self-assembly.

Reply with a single token on the first line: YES, NO, or BORDERLINE. Then on a second line provide a one-sentence rationale.

YES — primary subject is supramolecular chemistry as defined above.
NO — primary subject is biomedical (immunology, virology, cell biology, etc.), small-molecule chemistry without host-guest binding, or unrelated.
BORDERLINE — paper mentions supramolecular concepts but its primary focus is elsewhere.

Title: {title}

Abstract: {abstract}"""


Verdict = Literal["YES", "NO", "BORDERLINE", "PARSE_FAIL"]


def parse_verdict(text: str) -> tuple[str, str]:
    lines = (text or "").strip().splitlines()
    if not lines:
        return "PARSE_FAIL", ""
    first = lines[0].strip().upper().rstrip(".:;,")
    rationale = "\n".join(lines[1:]).strip()
    if first in ("YES", "NO", "BORDERLINE"):
        return first, rationale
    # Strict-only fallback: require the very first whitespace-delimited token
    # to be one of YES/NO/BORDERLINE; otherwise PARSE_FAIL.
    tokens = first.replace(",", " ").replace(".", " ").split()
    if tokens and tokens[0] in ("YES", "NO", "BORDERLINE"):
        return tokens[0], rationale or text.strip()[:240]
    return "PARSE_FAIL", text.strip()[:240]


async def judge_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    title: str,
    abstract: str,
) -> tuple[str, str]:
    prompt = JUDGE_USER_TEMPLATE.format(title=title[:2000], abstract=abstract[:6000])
    payload = {
        "model": MODEL,
        "max_tokens": 256,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.post(OPENROUTER_URL, json=payload, timeout=60.0)
                if resp.status_code == 429:
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"] or ""
                return parse_verdict(text)
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                if attempt == 2:
                    return "PARSE_FAIL", f"API_ERROR: {exc}"
                await asyncio.sleep(1.5 * (attempt + 1))
        return "PARSE_FAIL", "EXHAUSTED_RETRIES"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-split", type=int, default=SAMPLES_PER_SPLIT)
    parser.add_argument("--noise-floor-n", type=int, default=NOISE_FLOOR_N)
    parser.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip API calls; just sample and print sizes.")
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY not set; aborting.")

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load splits
    print(f"Loading dataset {DATASET_NAME} (subset={DATASET_SUBSET})...", file=sys.stderr)
    ds_raw = load_dataset(DATASET_NAME, DATASET_SUBSET, split="raw")
    ds_filt = load_dataset(DATASET_NAME, DATASET_SUBSET, split="filtered")
    print(f"  raw:      n={len(ds_raw)}  cols={list(ds_raw.features)}", file=sys.stderr)
    print(f"  filtered: n={len(ds_filt)}  cols={list(ds_filt.features)}", file=sys.stderr)

    def sample(ds, k):
        idxs = rng.sample(range(len(ds)), k=min(k, len(ds)))
        return [{"pmid": str(ds[i].get("id") or ds[i].get("pmid") or i),
                 "title": ds[i].get("title") or "",
                 "abstract": ds[i].get("abstract") or ""} for i in idxs]

    raw_samples = sample(ds_raw, args.samples_per_split)
    filt_samples = sample(ds_filt, args.samples_per_split)
    print(f"\nSampled raw: {len(raw_samples)}  filtered: {len(filt_samples)}", file=sys.stderr)

    filt_ids = {str(ds_filt[i].get("id")) for i in range(len(ds_filt))}
    excluded_in_raw_sample = [r for r in raw_samples if r["pmid"] not in filt_ids]
    print(f"  -> of {len(raw_samples)} raw samples, {len(excluded_in_raw_sample)} are excluded-only "
          f"(i.e., raw\\filtered)", file=sys.stderr)

    if args.dry_run:
        print("\nDry-run mode: skipping API calls.")
        return 0

    asyncio.run(judge_all(raw_samples, filt_samples, args))
    return 0


async def judge_all(raw_samples: list[dict], filt_samples: list[dict], args):
    api_key = os.environ["OPENROUTER_API_KEY"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Tianyi-Billy-Ma/SupraBench",
        "X-Title": "SupraBench Corpus Validation",
    }

    async with httpx.AsyncClient(headers=headers, timeout=60.0) as client:
        sem = asyncio.Semaphore(args.max_concurrency)

        async def one_call(rec):
            v, r = await judge_one(client, sem, rec["title"], rec["abstract"])
            return {"pmid": rec["pmid"], "title": rec["title"][:200],
                    "verdict": v, "rationale": r[:500]}

        # First pass
        print(f"\nFirst pass: {len(raw_samples) + len(filt_samples)} calls "
              f"(concurrency={args.max_concurrency})...", file=sys.stderr)
        start = time.monotonic()
        raw_results = await asyncio.gather(*[one_call(r) for r in raw_samples])
        print(f"  raw done in {time.monotonic()-start:.1f}s", file=sys.stderr)
        filt_results = await asyncio.gather(*[one_call(r) for r in filt_samples])
        print(f"  filtered done in {time.monotonic()-start:.1f}s", file=sys.stderr)

        # Save first-pass verdicts
        verdicts_path = OUT_DIR / "sample_verdicts.jsonl"
        with verdicts_path.open("w") as f:
            for r in raw_results:
                f.write(json.dumps({**r, "split": "raw"}, ensure_ascii=False) + "\n")
            for r in filt_results:
                f.write(json.dumps({**r, "split": "filtered"}, ensure_ascii=False) + "\n")
        print(f"  -> wrote {verdicts_path}", file=sys.stderr)

        # Noise-floor pass
        rng = random.Random(args.seed + 1)
        sample_lookup = {("raw", s["pmid"]): s for s in raw_samples}
        sample_lookup.update({("filtered", s["pmid"]): s for s in filt_samples})

        flat_first = [{**r, "split": "raw"} for r in raw_results] + \
                     [{**r, "split": "filtered"} for r in filt_results]
        nf_idxs = rng.sample(range(len(flat_first)), k=min(args.noise_floor_n, len(flat_first)))
        nf_first = [flat_first[i] for i in nf_idxs]
        nf_inputs = [sample_lookup[(r["split"], r["pmid"])] for r in nf_first]

        print(f"\nNoise-floor pass: re-judging {len(nf_inputs)} samples...", file=sys.stderr)
        nf_second = await asyncio.gather(*[one_call(r) for r in nf_inputs])

        noise_path = OUT_DIR / "noise_floor_verdicts.jsonl"
        agree = 0
        with noise_path.open("w") as f:
            for first, second in zip(nf_first, nf_second):
                agree += int(first["verdict"] == second["verdict"])
                f.write(json.dumps({
                    "pmid": first["pmid"], "split": first["split"],
                    "first_verdict": first["verdict"], "second_verdict": second["verdict"],
                    "first_rationale": first["rationale"], "second_rationale": second["rationale"],
                }, ensure_ascii=False) + "\n")
        print(f"  -> wrote {noise_path}", file=sys.stderr)
        print(f"  -> Haiku-Haiku agreement: {agree}/{len(nf_inputs)} = "
              f"{100*agree/len(nf_inputs):.1f}%", file=sys.stderr)

        # Summary
        def tally(results):
            c = {"YES": 0, "NO": 0, "BORDERLINE": 0, "PARSE_FAIL": 0}
            for r in results:
                c[r["verdict"]] = c.get(r["verdict"], 0) + 1
            return c

        raw_t = tally(raw_results)
        filt_t = tally(filt_results)

        summary = {
            "model": MODEL,
            "samples_per_split": args.samples_per_split,
            "seed": args.seed,
            "noise_floor_n": len(nf_inputs),
            "noise_floor_agreement": agree / len(nf_inputs) if nf_inputs else None,
            "raw": {
                "n": len(raw_results),
                "verdicts": raw_t,
                "yes_rate": raw_t.get("YES", 0) / len(raw_results),
                "borderline_rate": raw_t.get("BORDERLINE", 0) / len(raw_results),
                "no_rate": raw_t.get("NO", 0) / len(raw_results),
                "parse_fail_rate": raw_t.get("PARSE_FAIL", 0) / len(raw_results),
            },
            "filtered": {
                "n": len(filt_results),
                "verdicts": filt_t,
                "yes_rate": filt_t.get("YES", 0) / len(filt_results),
                "borderline_rate": filt_t.get("BORDERLINE", 0) / len(filt_results),
                "no_rate": filt_t.get("NO", 0) / len(filt_results),
                "parse_fail_rate": filt_t.get("PARSE_FAIL", 0) / len(filt_results),
            },
        }

        summary_path = OUT_DIR / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n  -> wrote {summary_path}", file=sys.stderr)

        # Stdout pretty-print
        print()
        print("=" * 70)
        print(f"  CORPUS VALIDATION SUMMARY  ({MODEL}, n={args.samples_per_split} per split)")
        print("=" * 70)
        print(f"{'verdict':>12s}  {'raw':>8s}  {'filtered':>10s}")
        for v in ("YES", "BORDERLINE", "NO", "PARSE_FAIL"):
            rp = 100 * raw_t.get(v, 0) / len(raw_results)
            fp = 100 * filt_t.get(v, 0) / len(filt_results)
            print(f"{v:>12s}  {rp:>6.1f}%  {fp:>9.1f}%")
        print()
        print(f"Haiku-Haiku self-agreement on {len(nf_inputs)}-sample re-judge: "
              f"{100*agree/len(nf_inputs):.1f}%")
        print()
        print(f"Files written to {OUT_DIR}/")


if __name__ == "__main__":
    raise SystemExit(main())
