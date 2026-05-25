"""
TBS · Step 01 — Data preparation.

Inputs:
  - suprabank.csv           Primary evaluation data
  - suprabank_smiles.csv    Guest structure dictionary (primary source)
  - CDEnrichedData.csv      Guest dict supplement + CD host SMILES + sanity subset

Outputs (data/processed/):
  - guest_smiles_dict.csv   Merged guest -> {iso_smiles, tags}
  - host_smiles_dict.csv    CD-family host -> iso_smiles
  - tbs_eval.parquet      Main evaluation set (~3772 rows, 6 named solvent labels)
  - cd_sanity.parquet       Full CDEnrichedData (3459 rows)

Usage:
    python scripts/sid_data_prep/01_data_prep.py [--root <repo_root>]

Identical to scripts/01_data_prep.py except for the --root default. SID
depends on its output data/processed/tbs_eval.parquet.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# Keep only the 6 named solvent classes; everything else (aromatics, long-tail
# polar solvents, mixtures) is dropped. "complex" mixtures get a second-chance
# remap via the `solvents` column further down.
SOLVENT_MAP: dict[str, str] = {
    "water": "water",
    "buffer": "water",
    "deuterium oxide": "water",
    "hydrochloric acid": "water",
    "sodium;chloride": "water",
    "dimethyl sulfoxide": "DMSO",
    "dimethyl sulfoxide-d6": "DMSO",
    "dmso-d6": "DMSO",
    "acetonitrile": "MeCN",
    "acetonitrile-d3": "MeCN",
    "methanol": "MeOH",
    "deuterated methanol": "MeOH",
    "chloroform": "CHCl3",
    "chloroform-d": "CHCl3",
    "methylene dichloride": "CH2Cl2",
    "dichloromethane-d2": "CH2Cl2",
    "cd2cl2": "CH2Cl2",
}


def map_solvent(raw: str | float) -> str | None:
    if not isinstance(raw, str):
        return None
    return SOLVENT_MAP.get(raw.lower().strip())


def host_family(h: str | float) -> str:
    if not isinstance(h, str):
        return "other"
    s = h.lower()
    if "cucurbit" in s or "cb[" in s:
        return "CB[n]"
    if "cyclodextrin" in s or "-cd" in s:
        return "cyclodextrin"
    if "sulfonat" in s and ("calix" in s or "arene" in s):
        return "sulfonato-calixarene"
    if "sulfonat" in s and "pillar" in s:
        return "sulfonato-pillararene"
    if "calix" in s:
        return "calixarene"
    if "pillar" in s:
        return "pillararene"
    if "cavitand" in s or "octa acid" in s:
        return "cavitand"
    if "naphthotube" in s:
        return "naphthotube"
    if "bambus" in s:
        return "bambusuril"
    if "zeolite" in s:
        return "zeolite"
    if "cryptand" in s:
        return "cryptand"
    if "metabasket" in s:
        return "metabasket"
    if "corona" in s or "crown" in s:
        return "crown_ether"
    if "cage" in s:
        return "cage"
    if "porphyrin" in s:
        return "porphyrin"
    if "cyclophane" in s:
        return "cyclophane"
    if "resorcin" in s:
        return "resorcinarene"
    return "other"


_GREEK_MAP = str.maketrans(
    {"α": "alpha", "β": "beta", "γ": "gamma", "Α": "alpha", "Β": "beta", "Γ": "gamma"}
)
_WS = re.compile(r"\s+")


def _norm(v) -> str | None:
    """Normalize host/guest names for join keys: lower + greek spelling + strip whitespace."""
    if not isinstance(v, str):
        return None
    k = _WS.sub("", v.translate(_GREEK_MAP).lower())
    return k or None


def build_guest_dict(sm: pd.DataFrame, cd: pd.DataFrame) -> pd.DataFrame:
    """Merge two sources; suprabank_smiles wins, CDEnrichedData only fills missing keys."""
    records: dict[str, dict[str, str | None]] = {}

    # Source 1: any of four name columns can serve as the join key
    for _, row in sm.iterrows():
        smi = row.get("iso_smiles") or row.get("cano_smiles")
        if not isinstance(smi, str) or not smi:
            continue
        tags = row.get("tags") if isinstance(row.get("tags"), str) else None
        for col in ("name", "iupac_name", "preferred_abbreviation", "molecule"):
            key = _norm(row.get(col))
            if key and key not in records:
                records[key] = {"smiles": smi, "tags": tags, "source": "suprabank_smiles"}

    # Source 2: fill missing only
    for _, row in cd.iterrows():
        key = _norm(row.get("Guest"))
        smi = row.get("IsomericSMILES")
        if not key or not isinstance(smi, str) or not smi:
            continue
        if key in records:
            continue
        records[key] = {"smiles": smi, "tags": None, "source": "CDEnrichedData"}

    return pd.DataFrame(
        [{"key": k, **v} for k, v in records.items()],
        columns=["key", "smiles", "tags", "source"],
    )


def build_host_smiles_dict(cd: pd.DataFrame) -> pd.DataFrame:
    sub = cd[["Host", "IsomericSMILES_Host"]].dropna().drop_duplicates(subset=["Host"])
    sub = sub.rename(columns={"Host": "host", "IsomericSMILES_Host": "host_smiles"})
    sub["key"] = sub["host"].map(_norm)
    return sub[["key", "host", "host_smiles"]].reset_index(drop=True)


def main(root: Path) -> int:
    sb_path = root / "suprabank.csv"
    sm_path = root / "suprabank_smiles.csv"
    cd_path = root / "CDEnrichedData.csv"
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    sb = pd.read_csv(sb_path)
    sm = pd.read_csv(sm_path)
    cd = pd.read_csv(cd_path)

    guest_dict = build_guest_dict(sm, cd)
    guest_dict.to_csv(out_dir / "guest_smiles_dict.csv", index=False)
    lookup = guest_dict.set_index("key")[["smiles", "tags"]]

    host_smiles = build_host_smiles_dict(cd)
    host_smiles.to_csv(out_dir / "host_smiles_dict.csv", index=False)
    host_lookup = host_smiles.set_index("key")["host_smiles"]

    sb = sb.copy()
    sb["source_row_idx"] = sb.index
    before = len(sb)
    eval_df = sb[sb["solvent"].notna()].copy()
    after = len(eval_df)
    print(f"[load] suprabank: {before} -> {after} rows (dropped {before - after} null-solvent)")

    eval_df["solvent_raw"] = eval_df["solvent"].astype(str)
    eval_df["solvent_label"] = eval_df["solvent_raw"].map(map_solvent)

    # Rescue: when raw solvent is "complex" but the `solvents` column names a
    # specific solvent, remap from there back to one of the 6 classes.
    rescue_mask = (
        eval_df["solvent_label"].isna()
        & (eval_df["solvent_raw"].str.lower().str.strip() == "complex")
        & eval_df["solvents"].notna()
    )
    rescued = eval_df.loc[rescue_mask, "solvents"].map(map_solvent)
    eval_df.loc[rescue_mask, "solvent_label"] = rescued
    n_rescued = rescued.notna().sum()
    print(f"[rescue] {n_rescued} 'complex' rows remapped via `solvents`")

    before_filter = len(eval_df)
    dropped_summary = (
        eval_df.loc[eval_df["solvent_label"].isna(), "solvent_raw"]
        .str.lower().str.strip().value_counts().head(10).to_dict()
    )
    eval_df = eval_df[eval_df["solvent_label"].notna()].reset_index(drop=True)
    print(
        f"[filter] 6-class filter: {before_filter} -> {len(eval_df)} rows "
        f"(dropped {before_filter - len(eval_df)}; top reasons: {dropped_summary})"
    )

    eval_df["host_family"] = eval_df["host"].map(host_family)

    eval_df["_host_key"] = eval_df["host"].map(_norm)
    eval_df["host_smiles"] = eval_df["_host_key"].map(host_lookup)

    eval_df["_guest_key"] = eval_df["molecule"].map(_norm)
    eval_df["guest_smiles"] = eval_df["_guest_key"].map(lookup["smiles"])
    eval_df["guest_tags"] = eval_df["_guest_key"].map(lookup["tags"])

    guest_match_rate = eval_df["guest_smiles"].notna().mean()
    print(f"[join] guest SMILES coverage: {guest_match_rate:.1%}")
    host_family_other_rate = (eval_df["host_family"] == "other").mean()
    print(f"[join] host_family == 'other' ratio: {host_family_other_rate:.1%}")

    eval_cols = [
        "interaction_id",
        "host",
        "host_family",
        "host_smiles",
        "molecule",
        "guest_smiles",
        "guest_tags",
        "solvent_raw",
        "solvent_label",
        "citation",
        "source_row_idx",
    ]
    eval_final = eval_df[eval_cols].rename(columns={"molecule": "guest"}).reset_index(drop=True)
    eval_final.to_parquet(out_dir / "tbs_eval.parquet", index=False)
    print(f"[save] tbs_eval.parquet: {len(eval_final)} rows")

    cd_sanity = cd.rename(
        columns={
            "Host": "host",
            "Guest": "guest",
            "IsomericSMILES": "guest_smiles",
            "IsomericSMILES_Host": "host_smiles",
        }
    ).copy()
    cd_sanity["host_family"] = cd_sanity["host"].map(host_family)
    cd_sanity["solvent_label"] = "water"
    cd_sanity_cols = [
        "host",
        "host_family",
        "host_smiles",
        "guest",
        "guest_smiles",
        "solvent_label",
        "Reference",
    ]
    cd_sanity[cd_sanity_cols].to_parquet(out_dir / "cd_sanity.parquet", index=False)
    print(f"[save] cd_sanity.parquet: {len(cd_sanity)} rows")

    print("\n=== solvent_label distribution ===")
    print(eval_final["solvent_label"].value_counts().to_string())
    print("\n=== host_family distribution ===")
    print(eval_final["host_family"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent.parent))
    args = ap.parse_args()
    sys.exit(main(Path(args.root)))
