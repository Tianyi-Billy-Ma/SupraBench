"""
Step 4 — Fill missing conditions with standard values.
Missing T  → assume 25.0 °C  (most common; standard lab condition)
Missing pH → assume 7.0       (neutral; most common value in dataset)

Adds columns:
  t_final       — temperature used (numeric °C)
  ph_final      — pH used (numeric)
  t_assumed     — True if T was missing and assumed
  ph_assumed    — True if pH was missing and assumed
Output: cleaned/04_filled.csv
"""

import csv
import os
import sys

IN  = "cleaned/03_aqueous.csv"
OUT = "cleaned/04_filled.csv"

DEFAULT_T  = 25.0
DEFAULT_PH = 7.0


def main():
    with open(IN, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assumed_t  = 0
    assumed_ph = 0

    for r in rows:
        # Temperature
        t_raw = r.get("t_numeric", "").strip()
        if t_raw:
            r["t_final"]    = t_raw
            r["t_assumed"]  = "False"
        else:
            r["t_final"]    = str(DEFAULT_T)
            r["t_assumed"]  = "True"
            assumed_t += 1

        # pH
        ph_raw = r.get("ph_numeric", "").strip()
        if ph_raw:
            r["ph_final"]   = ph_raw
            r["ph_assumed"] = "False"
        else:
            r["ph_final"]   = str(DEFAULT_PH)
            r["ph_assumed"] = "True"
            assumed_ph += 1

    all_keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(f"Filled {n} rows → {OUT}")
    print(f"  T  assumed = 25 °C : {assumed_t}/{n} ({100*assumed_t/n:.1f}%)")
    print(f"  pH assumed = 7.0   : {assumed_ph}/{n} ({100*assumed_ph/n:.1f}%)")


if __name__ == "__main__":
    main()
