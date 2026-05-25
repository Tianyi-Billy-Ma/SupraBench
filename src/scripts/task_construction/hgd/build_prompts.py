#!/usr/bin/env python3
"""Build prompts for the Host-Guest Description (HGD) task.

HGD has two complementary subtypes:
  forward : given a host, describe the molecular properties of guests that
            exhibit strong binding affinity (high log Ka) with it.
  reverse : given a guest (with SMILES), describe the molecular properties
            of hosts that exhibit strong binding affinity with it.

Both subtypes share the same source: the cleaned host-guest binding records
in ``all_standard.csv`` and the guest/host SMILES lookup dictionaries.

Output (one combined JSONL):
  benchmark/hgd.jsonl

Data dependencies (not included in this repository — populate before running):
  DATA_DIR/all_standard.csv         main binding-affinity table
  DATA_DIR/all_molecules_smiles.csv guest SMILES (primary)
  DATA_DIR/cb7_molecules_smiles.csv guest SMILES (supplement)
  DATA_DIR/host_smiles.csv          host SMILES

Usage:
  DATA_DIR=/path/to/data OUT_DIR=benchmark \
    python src/scripts/task_construction/hgd/build_prompts.py
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
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# Make `templates` importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))
from templates import generate_prompt  # noqa: E402

DATA_DIR = Path(os.environ.get("DATA_DIR", os.path.expanduser("~")))
OUT_DIR = Path(os.environ.get("OUT_DIR", REPO_ROOT / "benchmark"))

random.seed(42)
np.random.seed(42)

# ── HGD parameters ──────────────────────────────────────────────────────────
T3_MIN_GUESTS = 10    # minimum guests with SMILES per host (forward)
T3_TOP_FRACTION = 0.3  # top fraction considered "high affinity"
T3R_MIN_HOSTS = 5     # minimum distinct hosts per guest (reverse)


# ── Load cleaned binding data ───────────────────────────────────────────────
print("Loading all_standard.csv …")
df = pd.read_csv(DATA_DIR / "all_standard.csv", dtype=str)
df["logka"] = pd.to_numeric(df["logka_standard"], errors="coerce")
df["molecule"] = df["molecule"].str.strip()
df["host"] = df["host"].str.strip()
valid = df[df["logka"].notna()].copy().reset_index(drop=True)
print(f"  Total rows: {len(df)},  valid (logKa numeric): {len(valid)}")

# ── SMILES lookups ──────────────────────────────────────────────────────────
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
            s = str(r.get(smi_col, "")).strip()
            if n and s and s not in ("nan", ""):
                name_to_smi[n] = s
    except FileNotFoundError:
        print(f"  Warning: {path} not found")
print(f"  Guest SMILES entries loaded: {len(name_to_smi)}")

host_to_smi: dict[str, str] = {}
try:
    tmp = pd.read_csv(DATA_DIR / "host_smiles.csv", dtype=str)
    for _, r in tmp.iterrows():
        n = str(r.get("host", "")).strip()
        s = str(r.get("smiles", "")).strip()
        if n and s and s not in ("nan", ""):
            host_to_smi[n] = s
    print(f"  Host SMILES entries loaded: {len(host_to_smi)}")
except FileNotFoundError:
    print("  Warning: host_smiles.csv not found")


def host_smi_str(host_name: str) -> str:
    s = host_to_smi.get(host_name, "")
    return f" (SMILES: {s})" if s else ""


def compute_props(smiles: str) -> dict | None:
    """RDKit-derived guest properties used to summarize 'high-affinity' guests."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return {
            "mw":     round(Descriptors.MolWt(mol), 1),
            "hbd":    rdMolDescriptors.CalcNumHBD(mol),
            "hba":    rdMolDescriptors.CalcNumHBA(mol),
            "rotb":   rdMolDescriptors.CalcNumRotatableBonds(mol),
            "rings":  rdMolDescriptors.CalcNumRings(mol),
            "charge": Chem.GetFormalCharge(mol),
        }
    except Exception:
        return None


def classify_host(host_name: str) -> str:
    """Brief qualitative class label inferred from the host's textual name."""
    h = host_name.lower()
    if "cucurbit" in h:
        return "cucurbituril macrocycle (barrel-shaped, hydrophobic cavity, carbonyl-lined portals)"
    if "cyclodextrin" in h or "-cd" in h or "cd)" in h:
        return "cyclodextrin oligosaccharide (truncated-cone shape, hydrophobic interior)"
    if "calix" in h:
        return "calixarene macrocycle (cone-shaped, phenol-derived)"
    if "pillar" in h:
        return "pillar[n]arene macrocycle (cylindrical, electron-rich cavity)"
    if "crown" in h:
        return "crown ether macrocycle (cyclic polyether)"
    if "cavitand" in h or "octa acid" in h:
        return "cavitand (deep hydrophobic pocket)"
    if "tweezer" in h:
        return "molecular tweezer (open-ended cleft receptor)"
    if "naph" in h and ("tube" in h or "tub" in h):
        return "naphthotube macrocycle (aromatic cylindrical cavity)"
    return "macrocyclic host molecule"


# ─────────────────────────────────────────────────────────────────────────────
# Forward subtype — describe high-affinity guest properties given a host
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating HGD (forward) questions …")

TASK_FORWARD_QUERY = (
    "For the host molecule {host}{host_smi}, describe the key molecular "
    "properties of guest molecules that exhibit strong binding affinity "
    "(high logKa) in aqueous solution at standard conditions. Include: "
    "approximate molecular weight range, typical formal charge, number of "
    "ring systems, and any characteristic functional groups."
)

records: list[dict] = []

for host_name, grp in valid.groupby("host"):
    dedup = grp.drop_duplicates(subset="molecule").copy()
    dedup["smiles"] = dedup["molecule"].map(name_to_smi).fillna("")
    dedup_smi = dedup[dedup["smiles"] != ""].copy()
    if len(dedup_smi) < T3_MIN_GUESTS:
        continue

    props_rows = []
    for _, r in dedup_smi.iterrows():
        p = compute_props(r["smiles"])
        if p:
            p["molecule"] = r["molecule"]
            p["logka"] = r["logka"]
            props_rows.append(p)
    if len(props_rows) < T3_MIN_GUESTS:
        continue

    pdf = pd.DataFrame(props_rows)
    threshold = pdf["logka"].quantile(1 - T3_TOP_FRACTION)
    top = pdf[pdf["logka"] >= threshold]

    mw_mean = top["mw"].mean()
    mw_std = top["mw"].std()
    charge_counts = top["charge"].value_counts()
    dominant_charge = int(charge_counts.index[0])
    avg_rings = top["rings"].mean()
    avg_hbd = top["hbd"].mean()
    avg_hba = top["hba"].mean()

    if dominant_charge > 0:
        charge_desc = "positively charged"
    elif dominant_charge < 0:
        charge_desc = "negatively charged"
    else:
        charge_desc = "neutral"

    gt = (
        f"High-affinity guests of {host_name} (logKa ≥ {threshold:.1f}) "
        f"typically have a molecular weight of {mw_mean:.0f}±{mw_std:.0f} g/mol, "
        f"are {charge_desc} (formal charge {dominant_charge:+d}), "
        f"contain on average {avg_rings:.1f} ring system(s), "
        f"{avg_hbd:.1f} H-bond donor(s), and {avg_hba:.1f} H-bond acceptor(s). "
        f"Representative guests: {', '.join(top['molecule'].head(5).tolist())}."
    )

    query = TASK_FORWARD_QUERY.format(host=host_name, host_smi=host_smi_str(host_name))
    records.append({
        "id":            f"hgd_fwd_{host_name.replace(' ', '_')}",
        "task":          "hgd",
        "subtype":       "forward",
        "question":      generate_prompt(query),
        "answer":        gt,
        "host_name":     host_name,
        "n_guests_smi":  len(dedup_smi),
        "n_top":         len(top),
        "gt_mw_mean":    round(mw_mean, 1),
        "gt_mw_std":     round(mw_std, 1),
        "gt_charge":     dominant_charge,
        "gt_rings_mean": round(avg_rings, 2),
    })

print(f"  HGD forward questions: {sum(1 for r in records if r['subtype'] == 'forward')}")


# ─────────────────────────────────────────────────────────────────────────────
# Reverse subtype — describe high-affinity host properties given a guest
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating HGD (reverse) questions …")

TASK_REVERSE_QUERY = (
    "For the guest molecule {guest} (SMILES: {smiles}), describe the key "
    "molecular properties of host molecules that exhibit strong binding "
    "affinity (high logKa) in aqueous solution at standard conditions. "
    "Include: host structural type, approximate cavity size or geometry, "
    "typical charge, and any characteristic binding interactions."
)

n_reverse = 0
for guest_name, grp in valid.groupby("molecule"):
    smiles = name_to_smi.get(guest_name, "")
    if not smiles:
        continue

    hosts_for_guest = grp.drop_duplicates(subset="host")[["host", "logka"]].copy()
    if len(hosts_for_guest) < T3R_MIN_HOSTS:
        continue

    hosts_for_guest = hosts_for_guest.sort_values("logka", ascending=False).reset_index(drop=True)
    threshold = hosts_for_guest["logka"].quantile(1 - T3_TOP_FRACTION)
    top_hosts = hosts_for_guest[hosts_for_guest["logka"] >= threshold]

    max_logka = float(hosts_for_guest["logka"].max())
    top_names = top_hosts["host"].tolist()
    top_logkas = top_hosts["logka"].tolist()

    host_classes = list(dict.fromkeys(classify_host(h) for h in top_names))
    class_desc = "; ".join(host_classes[:3])

    top_entries = ", ".join(
        f"{h} (logKa={lk:.1f})" for h, lk in zip(top_names[:5], top_logkas[:5])
    )

    gt = (
        f"High-affinity hosts for {guest_name} (logKa ≥ {threshold:.1f}) "
        f"include: {top_entries}. "
        f"The highest recorded logKa is {max_logka:.1f}. "
        f"Favorable hosts are predominantly {class_desc}."
    )

    query = TASK_REVERSE_QUERY.format(guest=guest_name, smiles=smiles)
    records.append({
        "id":          f"hgd_rev_{guest_name.replace(' ', '_')}",
        "task":        "hgd",
        "subtype":     "reverse",
        "question":    generate_prompt(query),
        "answer":      gt,
        "guest_name":  guest_name,
        "guest_smiles": smiles,
        "n_hosts":     len(hosts_for_guest),
        "n_top_hosts": len(top_hosts),
        "max_logka":   round(max_logka, 4),
    })
    n_reverse += 1

print(f"  HGD reverse questions: {n_reverse}")
print(f"  HGD total: {len(records)}")


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "hgd.jsonl"
with out_path.open("w") as fh:
    for q in records:
        fh.write(json.dumps(q, ensure_ascii=False) + "\n")
print(f"\nSaved: {out_path}  ({len(records)} lines)")
