"""
Step 1 — Merge raw datasets.
Combines cb7_full.csv and all_interactions.csv into one raw file.
Output: cleaned/01_merged.csv
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

RAW_FILES = [
    "../cb7_full.csv",
    "../all_interactions.csv",
]
OUT = "cleaned/01_merged.csv"


def main():
    os.makedirs("cleaned", exist_ok=True)
    all_rows = []
    seen_ids = set()

    for path in RAW_FILES:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                iid = row.get("interaction_id", "")
                if iid and iid in seen_ids:
                    continue          # deduplicate
                seen_ids.add(iid)
                all_rows.append(row)

    if not all_rows:
        print("No data found.")
        return

    all_keys = list(dict.fromkeys(k for r in all_rows for k in r.keys()))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print(f"Merged {len(all_rows)} rows → {OUT}")


if __name__ == "__main__":
    main()
