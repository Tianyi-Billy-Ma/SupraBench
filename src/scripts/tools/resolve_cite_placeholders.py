"""Resolve `[CITE: short-description]` placeholders in paper sections against
the actual citation keys defined in `paper/custom.bib`.

Pattern-based mapping. Each rule pairs a regex against the placeholder text
with a bibtex key that already exists in custom.bib. Unmatched placeholders
are left in place so the user can fill them in manually.

Reports stats: how many placeholders existed, how many were resolved,
how many remain.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper"
BIB = PAPER / "custom.bib"
SECTIONS = [
    PAPER / "sections" / "benchmark.tex",
    PAPER / "sections" / "experiments.tex",
    PAPER / "sections" / "appendix.tex",
]

# Load known keys from custom.bib
known_keys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text()))

# Mapping rules: (regex on placeholder text -> bibtex key)
# Match keywords case-insensitively. First-match wins.
RULES = [
    (r"ChemAgent",                         "tang2025chemagent"),
    (r"ChemBench",                         "mirza2025chembench"),
    (r"ChemCrow",                          "bran2024chemcrow"),
    (r"ChemDFM",                           "zhao2025chemdfm"),
    (r"ChemLLMBench",                      "guo2023chemllmbench"),
    (r"\bChemLLM\b(?!Bench)",              "zhang2024chemllm"),
    (r"Coscientist",                       "boiko2023coscientist"),
    (r"Galactica",                         "taylor2022galactica"),
    (r"GuacaMol",                          "brown2019guacamol"),
    (r"LabBench",                          "laurent2024labbench"),
    (r"\bLlaSMol\b|\bLlasmol\b",           "yu2024llasmol"),
    (r"Mol-Instructions|MolInstructions",  "fang2024molinstructions"),
    (r"MoleculeNet",                       "wu2018moleculenet"),
    (r"\bMolT5\b",                         "edwards2022molt5"),
    (r"\bMOSES\b",                         "polykovskiy2020moses"),
    (r"Nach0",                             "livne2024nach0"),
    (r"SAMPL8|SAMPL host-guest|Amezcua",   "amezcua2022sampl8"),
    (r"SciBench",                          "wang2024scibench"),
    (r"SciEval",                           "sun2024scieval"),
    (r"SupraML|Colaco",                    "colaco2024supraml"),
    (r"\bTDC\b",                           "huang2021tdc"),
    (r"Lehn supramolecular|Lehn .*\boverview\b",        "lehn1995supramolecular"),
    (r"Steed|supramolecular .*textbook",                "steed2009supramolecular"),
    (r"Ariga|molecular[- ]recognition .*review",        "ariga2012molecular"),
    (r"sugammadex|Bom et al",                           "bom2002sugammadex"),
    (r"cyclodextrin .*review|Loftsson",                 "loftsson2010cyclodextrins"),
    (r"webber|drug delivery .*review .*2017",           "webber2017drug"),
    (r"pillar.*6.*arene|pillar\[6\]arene|Brockett",     "brockett2023pillar6maxq"),
    (r"Deng .*sequestration|in[- ]vivo sequestration",  "deng2020sequestration"),
    (r"\bYou et al\b|host-guest sensing review",        "you2015sensing"),
    (r"Kolesnichenko|practical .*supramolecular",       "kolesnichenko2017practical"),
]

# Sanity-check that every rule key is actually in the bib.
for _, key in RULES:
    if key not in known_keys:
        raise RuntimeError(f"rule references unknown key {key!r}; not in custom.bib")

CITE_RE = re.compile(r"\[CITE:\s*([^\]]+?)\s*\]")

def resolve(placeholder_text: str) -> str | None:
    for pattern, key in RULES:
        if re.search(pattern, placeholder_text, flags=re.IGNORECASE):
            return key
    return None

def process(path: Path) -> tuple[int, int]:
    text = path.read_text()
    placeholders_before = CITE_RE.findall(text)
    resolved = 0
    def repl(m: re.Match) -> str:
        nonlocal resolved
        desc = m.group(1)
        key = resolve(desc)
        if key is None:
            return m.group(0)
        resolved += 1
        return f"\\citep{{{key}}}"
    new_text = CITE_RE.sub(repl, text)
    if new_text != text:
        path.write_text(new_text)
    return len(placeholders_before), resolved

for f in SECTIONS:
    if not f.is_file():
        continue
    n, r = process(f)
    remaining = n - r
    print(f"{f.name:30s}  placeholders={n:3d}  resolved={r:3d}  remaining={remaining:3d}")

# Final sanity: still no invented keys
import subprocess
print()
print("=== post-replacement \\cite usage scan ===")
all_used = set()
for f in SECTIONS:
    if not f.is_file(): continue
    for m in re.finditer(r"\\cite[a-z]*\{([^}]+)\}", f.read_text()):
        for k in m.group(1).split(","):
            all_used.add(k.strip())
missing = sorted(all_used - known_keys)
print(f"distinct \\cite keys used across all 3 sections: {len(all_used)}")
print(f"missing-from-bib keys (must be 0): {missing}")
if missing:
    raise SystemExit("ERROR: introduced cite keys not in bib")
print("all keys verified against custom.bib")
