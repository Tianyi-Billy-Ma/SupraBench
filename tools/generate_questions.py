#!/usr/bin/env python3
"""Generate Task 1 and Task 2 benchmark questions from the raw dataset.

  Task 1 : logKa prediction (numerical)
  Task 2 : 4-choice MCQ — which guest has the highest binding affinity?

Each task produces three prompt variants:
  base    — plain query only
  fewshot — plain query + 3 few-shot examples
  cot     — plain query + 3 few-shot examples + chain-of-thought cue

Output (relative to repo root):
  data/task1/base.jsonl
  data/task1/fewshot.jsonl
  data/task1/cot.jsonl
  data/task2/base.jsonl
  data/task2/fewshot.jsonl
  data/task2/cot.jsonl

Data dependencies (not included in this repo):
  DATA_DIR/all_standard.csv        — main binding-affinity table
  DATA_DIR/all_molecules_smiles.csv — guest SMILES
  DATA_DIR/cb7_molecules_smiles.csv — additional guest SMILES
  DATA_DIR/host_smiles.csv          — host SMILES

Usage:
  DATA_DIR=/path/to/data python tools/generate_questions.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from templates import generate_options, generate_prompt  # noqa: E402

DATA_DIR = Path(os.environ.get("DATA_DIR", os.path.expanduser("~")))
OUT_DIR  = REPO_ROOT / "data"

# ── Task 2 parameters ─────────────────────────────────────────────────────────
N_CHOICES    = 4
MIN_SPREAD   = 0.5
MAX_PER_HOST = 200

random.seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading all_standard.csv …")
df = pd.read_csv(DATA_DIR / "all_standard.csv", dtype=str)
df["logka"]    = pd.to_numeric(df["logka_standard"], errors="coerce")
df["molecule"] = df["molecule"].str.strip()
df["host"]     = df["host"].str.strip()
valid = df[df["logka"].notna()].copy().reset_index(drop=True)
print(f"  Total rows: {len(df)},  valid (logKa numeric): {len(valid)}")

# ── SMILES lookup ─────────────────────────────────────────────────────────────
print("Loading SMILES maps …")
name_to_smi: dict[str, str] = {}
for path, name_col, smi_col in [
    (DATA_DIR / "all_molecules_smiles.csv", "molecule", "cano_smiles"),
    (DATA_DIR / "cb7_molecules_smiles.csv", "molecule", "cano_smiles"),
]:
    try:
        tmp = pd.read_csv(path, dtype=str)
        for _, r in tmp.iterrows():
            n = str(r.get(name_col, "")).strip()
            s = str(r.get(smi_col,  "")).strip()
            if n and s and s not in ("nan", ""):
                name_to_smi[n] = s
    except FileNotFoundError:
        print(f"  Warning: {path} not found")
print(f"  Guest SMILES entries loaded: {len(name_to_smi)}")

host_to_smi: dict[str, str] = {}
try:
    tmp = pd.read_csv(DATA_DIR / "host_smiles.csv", dtype=str)
    for _, r in tmp.iterrows():
        n = str(r.get("host",   "")).strip()
        s = str(r.get("smiles", "")).strip()
        if n and s and s not in ("nan", ""):
            host_to_smi[n] = s
    print(f"  Host SMILES entries loaded: {len(host_to_smi)}")
except FileNotFoundError:
    print("  Warning: host_smiles.csv not found")


def smi_str(mol_name: str) -> str:
    s = name_to_smi.get(mol_name, "")
    return f" (SMILES: {s})" if s else ""


def host_smi_str(host_name: str) -> str:
    s = host_to_smi.get(host_name, "")
    return f" (SMILES: {s})" if s else ""


# ─────────────────────────────────────────────────────────────────────────────
# 2. Task 1 — logKa prediction
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating Task 1 questions …")

TASK1_BASE = (
    "Predict the binding affinity (logKa) between the host molecule "
    "{host}{host_smi} and the guest molecule {guest}{guest_smi} in aqueous solution at "
    "standard conditions. Provide a single numerical value."
)

_pool_t1 = (
    valid[valid["molecule"].isin(name_to_smi) & valid["host"].isin(host_to_smi)]
    .sort_values("logka")
    .reset_index(drop=True)
)
_n1       = len(_pool_t1)
_fs1_rows = _pool_t1.iloc[[0, _n1 // 2, _n1 - 1]]
_fs1_ids  = set(_fs1_rows["interaction_ids"].astype(str))

FEWSHOT_T1 = [
    {
        "query": TASK1_BASE.format(
            host=r["host"],      host_smi=host_smi_str(r["host"]),
            guest=r["molecule"], guest_smi=smi_str(r["molecule"]),
        ),
        "answer": str(round(float(r["logka"]), 1)),
    }
    for _, r in _fs1_rows.iterrows()
]
print(f"  Few-shot logKa values: {[ex['answer'] for ex in FEWSHOT_T1]}")

task1_base, task1_fewshot, task1_cot = [], [], []

for _, row in valid.iterrows():
    if str(row["interaction_ids"]) in _fs1_ids:
        continue
    guest = row["molecule"]
    query = TASK1_BASE.format(
        host=row["host"],     host_smi=host_smi_str(row["host"]),
        guest=guest,          guest_smi=smi_str(guest),
    )
    rec = {
        "id":        f"t1_{row['interaction_ids']}",
        "task":      "task1",
        "answer":    round(float(row["logka"]), 4),
        "host_name": row["host"],
        "molecule":  guest,
    }
    task1_base.append(   {**rec, "version": "v1", "question": generate_prompt(query)})
    task1_fewshot.append({**rec, "version": "v2", "question": generate_prompt(query, fewshot_examples=FEWSHOT_T1)})
    task1_cot.append(    {**rec, "version": "v3", "question": generate_prompt(query, fewshot_examples=FEWSHOT_T1, thinking=True)})

print(f"  Task 1 per variant: {len(task1_base)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Task 2 — 4-choice MCQ
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating Task 2 questions …")

TASK2_STEM = (
    "Which of the following guest molecules has the strongest binding "
    "affinity (highest logKa) with the host molecule {host}{host_smi} in aqueous "
    "solution at standard conditions?"
)

FEWSHOT_T2: list[dict] = []
_fs2_excluded: dict[str, frozenset] = {}

for host_name in sorted(valid["host"].unique()):
    if len(FEWSHOT_T2) >= 3:
        break
    grp   = valid[valid["host"] == host_name]
    dedup = grp.drop_duplicates(subset="molecule").reset_index(drop=True)
    if len(dedup) < N_CHOICES:
        continue
    molecules = dedup["molecule"].tolist()
    logkas    = dedup["logka"].tolist()
    rng_fs    = np.random.RandomState(abs(hash(host_name + "_fs")) % (2**31))
    for _ in range(200):
        idx   = rng_fs.choice(len(dedup), size=N_CHOICES, replace=False)
        lk    = [logkas[i] for i in idx]
        if max(lk) - min(lk) < MIN_SPREAD:
            continue
        disp   = rng_fs.permutation(N_CHOICES).tolist()
        g_disp = [molecules[idx[j]] for j in disp]
        l_disp = [lk[j] for j in disp]
        best   = int(np.argmax(l_disp))
        letter = chr(ord("A") + best)
        opts   = [f"{g}{smi_str(g)}" for g in g_disp]
        q_ex   = (TASK2_STEM.format(host=host_name, host_smi=host_smi_str(host_name))
                  + "\n\n" + generate_options(opts))
        FEWSHOT_T2.append({"query": q_ex, "answer": letter})
        _fs2_excluded[host_name] = frozenset(idx.tolist())
        break

print(f"  Few-shot hosts: {list(_fs2_excluded.keys())}")

task2_base, task2_fewshot, task2_cot = [], [], []

for host_name, grp in valid.groupby("host"):
    dedup    = grp.drop_duplicates(subset="molecule").reset_index(drop=True)
    n_guests = len(dedup)
    if n_guests < N_CHOICES:
        continue
    n_qs      = min(MAX_PER_HOST, n_guests)
    molecules = dedup["molecule"].tolist()
    logkas    = dedup["logka"].tolist()
    rng       = np.random.RandomState(abs(hash(host_name)) % (2**31))
    excl_key  = _fs2_excluded.get(host_name)
    generated = 0
    seen_sets: set[frozenset] = set()
    for _ in range(n_qs * 10):
        if generated >= n_qs:
            break
        chosen_idx = rng.choice(n_guests, size=N_CHOICES, replace=False)
        key = frozenset(chosen_idx.tolist())
        if key in seen_sets or key == excl_key:
            continue
        seen_sets.add(key)
        lk_vals = [logkas[i] for i in chosen_idx]
        if max(lk_vals) - min(lk_vals) < MIN_SPREAD:
            continue
        display_order   = list(range(N_CHOICES))
        random.shuffle(display_order)
        guests_disp     = [molecules[chosen_idx[j]] for j in display_order]
        lk_disp         = [lk_vals[j] for j in display_order]
        best_in_display = int(np.argmax(lk_disp))
        correct_letter  = chr(ord("A") + best_in_display)
        options_with_smi = [f"{g}{smi_str(g)}" for g in guests_disp]
        query = (TASK2_STEM.format(host=host_name, host_smi=host_smi_str(host_name))
                 + "\n\n" + generate_options(options_with_smi))
        rec = {
            "id":               f"t2_{host_name}_{generated}",
            "task":             "task2",
            "answer":           correct_letter,
            "correct_molecule": guests_disp[best_in_display],
            "options":          guests_disp,
            "options_logka":    [round(v, 4) for v in lk_disp],
            "host_name":        host_name,
        }
        task2_base.append(   {**rec, "version": "v1", "question": generate_prompt(query)})
        task2_fewshot.append({**rec, "version": "v2", "question": generate_prompt(query, fewshot_examples=FEWSHOT_T2)})
        task2_cot.append(    {**rec, "version": "v3", "question": generate_prompt(query, fewshot_examples=FEWSHOT_T2, thinking=True)})
        generated += 1

print(f"  Task 2 per variant: {len(task2_base)}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Save
# ─────────────────────────────────────────────────────────────────────────────
(OUT_DIR / "task1").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "task2").mkdir(parents=True, exist_ok=True)

for path, data in [
    (OUT_DIR / "task1" / "base.jsonl",    task1_base),
    (OUT_DIR / "task1" / "fewshot.jsonl", task1_fewshot),
    (OUT_DIR / "task1" / "cot.jsonl",     task1_cot),
    (OUT_DIR / "task2" / "base.jsonl",    task2_base),
    (OUT_DIR / "task2" / "fewshot.jsonl", task2_fewshot),
    (OUT_DIR / "task2" / "cot.jsonl",     task2_cot),
]:
    with open(path, "w") as f:
        for q in data:
            f.write(json.dumps(q) + "\n")
    print(f"Saved: {path}  ({len(data)} lines)")
