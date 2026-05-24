"""
Step 5 — Temperature correction to 25 °C via van't Hoff equation.

ln(Ka₂/Ka₁) = −ΔH°/R × (1/T₂ − 1/T₁)

Problem: The Suprabank dataset does not contain reliable ΔH values.
Strategy:
  1. Rows already at 25 °C → pass through unchanged.
  2. Rows at T ≠ 25 °C with a valid ka_numeric:
       a. If a ΔH value is ever available (future data), use it.
       b. Otherwise: use a HOST-SPECIFIC default ΔH from the literature
          for the most common hosts (CB7, CB8, β-CD, γ-CD, etc.).
          If no default available → flag as 'correction_not_possible' and
          keep the original Ka with a warning column.

Adds columns:
  ka_25c          — Ka corrected to 25 °C (or original if already at 25 °C)
  logka_25c       — log10(ka_25c)
  t_correction    — 'none' / 'vanthoff' / 'assumed_25c' / 'not_possible'
Output: cleaned/05_t_corrected.csv
"""

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_cleaning.utils import vanthoff_correct_ka, R, T_STD

IN  = "cleaned/04_filled.csv"
OUT = "cleaned/05_t_corrected.csv"

# Literature ΔH° (kJ/mol) for common hosts — used when no measured ΔH available.
# Negative = exothermic (typical for cucurbiturils, cyclodextrins).
# Values are representative means from published thermodynamic datasets.
HOST_DH_DEFAULTS = {
    # Cucurbiturils
    "cucurbit[7]uril":  -40.0e3,   # ~−40 kJ/mol typical for CB7
    "cucurbit[8]uril":  -35.0e3,
    "cucurbit[6]uril":  -25.0e3,
    "cucurbit[5]uril":  -20.0e3,
    # Cyclodextrins
    "β-cyclodextrin":   -20.0e3,
    "alpha-cyclodextrin": -15.0e3,
    "gamma-cyclodextrin": -18.0e3,
    # Calixarenes (less certain)
    "p-sulfonatocalix[4]arene": -30.0e3,
}

def get_default_dh(host: str):
    """Match host name (case-insensitive) to default ΔH, return J/mol or None."""
    h = host.lower().strip()
    for key, dh in HOST_DH_DEFAULTS.items():
        if key in h or h in key:
            return dh
    return None


def main():
    with open(IN, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    stats = {"none": 0, "vanthoff": 0, "assumed_25c": 0, "not_possible": 0}

    for r in rows:
        try:
            ka = float(r.get("ka_numeric", "") or "")
        except ValueError:
            # No numeric Ka — can't correct; just carry forward
            r["ka_25c"]       = r.get("ka_numeric", "")
            r["logka_25c"]    = r.get("logka_numeric", "")
            r["t_correction"] = "not_possible"
            stats["not_possible"] += 1
            continue

        try:
            t = float(r.get("t_final", "") or "")
        except ValueError:
            t = 25.0

        assumed = r.get("t_assumed", "False") == "True"

        if assumed:
            # Temperature was assumed to be 25 °C — no correction needed
            r["ka_25c"]       = str(ka)
            r["logka_25c"]    = f"{math.log10(ka):.4f}"
            r["t_correction"] = "assumed_25c"
            stats["assumed_25c"] += 1

        elif abs(t - 25.0) < 0.1:
            # Already at 25 °C
            r["ka_25c"]       = str(ka)
            r["logka_25c"]    = f"{math.log10(ka):.4f}"
            r["t_correction"] = "none"
            stats["none"] += 1

        else:
            # Need correction — look for ΔH
            dh = get_default_dh(r.get("host", ""))
            if dh is not None:
                ka_25 = vanthoff_correct_ka(ka, t, dh)
                r["ka_25c"]       = f"{ka_25:.6g}"
                r["logka_25c"]    = f"{math.log10(ka_25):.4f}"
                r["t_correction"] = "vanthoff"
                stats["vanthoff"] += 1
            else:
                # No ΔH available — keep original, flag
                r["ka_25c"]       = str(ka)
                r["logka_25c"]    = f"{math.log10(ka):.4f}"
                r["t_correction"] = "not_possible"
                stats["not_possible"] += 1

    all_keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(f"Temperature correction → {OUT}  ({n} rows)")
    print(f"  Already 25 °C (no correction): {stats['none']}")
    print(f"  Corrected via van't Hoff:       {stats['vanthoff']}")
    print(f"  T assumed 25 °C:                {stats['assumed_25c']}")
    print(f"  Could not correct (no ΔH):      {stats['not_possible']}")
    print()
    print("  NOTE: van't Hoff correction uses literature ΔH° defaults.")
    print("  Rows with t_correction='not_possible' have T≠25°C but no ΔH.")
    print("  These are kept in the dataset but flagged — exclude if needed.")


if __name__ == "__main__":
    main()
