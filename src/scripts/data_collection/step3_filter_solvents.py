"""
Step 3 — Filter organic solvents.
Adds 'solvent_class' column (aqueous / organic / unknown).
Drops rows with solvent_class == 'organic'.
Output: cleaned/03_aqueous.csv  +  cleaned/03_dropped_organic.csv
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_cleaning.utils import classify_solvent

IN   = "cleaned/02_parsed.csv"
OUT  = "cleaned/03_aqueous.csv"
DROP = "cleaned/03_dropped_organic.csv"


def main():
    with open(IN, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        r["solvent_class"] = classify_solvent(r.get("solvent", ""), r.get("solvents", ""))

    aqueous  = [r for r in rows if r["solvent_class"] != "organic"]
    organic  = [r for r in rows if r["solvent_class"] == "organic"]

    all_keys = list(dict.fromkeys(k for r in rows for k in r.keys()))

    for path, data in [(OUT, aqueous), (DROP, organic)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(data)

    print(f"Total:   {len(rows)}")
    print(f"Kept (aqueous/unknown): {len(aqueous)} → {OUT}")
    print(f"Dropped (organic):      {len(organic)} → {DROP}")

    # Breakdown of unknown
    unknown = [r for r in aqueous if r["solvent_class"] == "unknown"]
    print(f"  (of kept: {len(unknown)} marked 'unknown' — inspect if needed)")


if __name__ == "__main__":
    main()
