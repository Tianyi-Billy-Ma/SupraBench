"""
Step 7 — Outlier removal.

Applied to the PRE-averaged data (05_t_corrected.csv) per (molecule, host) group,
to identify and remove anomalous Ka measurements before averaging.

Method: Tukey IQR fence on logka_25c values within each (molecule, host) group.
  - Groups with < 4 measurements: no outlier removal (too few points).
  - Outlier threshold: k = 1.5 × IQR  (standard), or 3.0 × IQR (conservative).

Two output files:
  cleaned/07_no_outliers.csv    — rows with outliers removed (feeds back into step6)
  cleaned/07_outliers.csv       — the flagged outlier rows
"""

import csv
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_cleaning.utils import iqr_outlier_mask

IN_PRE_AVG = "cleaned/05_t_corrected.csv"   # pre-average data
OUT_CLEAN  = "cleaned/07_no_outliers.csv"
OUT_FLAGGED = "cleaned/07_outliers.csv"

IQR_K = 1.5   # standard Tukey fence multiplier


def main():
    with open(IN_PRE_AVG, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Build (molecule, host) groups — only rows with valid logka_25c
    groups = defaultdict(list)
    no_ka  = []
    for i, r in enumerate(rows):
        lk = r.get("logka_25c", "").strip()
        if lk:
            try:
                float(lk)
                groups[(r.get("molecule",""), r.get("host",""))].append(i)
            except ValueError:
                no_ka.append(i)
        else:
            no_ka.append(i)

    outlier_indices = set()
    group_stats = []

    for (mol, host), indices in groups.items():
        logka_vals = [float(rows[i]["logka_25c"]) for i in indices]
        mask = iqr_outlier_mask(logka_vals, k=IQR_K)
        n_out = sum(mask)
        if n_out:
            group_stats.append((mol, host, len(indices), n_out,
                                 min(logka_vals), max(logka_vals)))
        for i, is_out in zip(indices, mask):
            if is_out:
                outlier_indices.add(i)

    clean   = [r for i, r in enumerate(rows) if i not in outlier_indices]
    flagged = [r for i, r in enumerate(rows) if i in outlier_indices]

    all_keys = list(dict.fromkeys(k for r in rows for k in r.keys()))

    for path, data in [(OUT_CLEAN, clean), (OUT_FLAGGED, flagged)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(data)

    print(f"Input rows:   {len(rows)}")
    print(f"Outliers:     {len(flagged)}  (IQR k={IQR_K}, min group size = 4)")
    print(f"Clean rows:   {len(clean)} → {OUT_CLEAN}")
    print(f"Flagged rows: {len(flagged)} → {OUT_FLAGGED}")

    if group_stats:
        print(f"\nGroups with outliers removed ({len(group_stats)}):")
        for mol, host, n_total, n_out, lk_min, lk_max in sorted(group_stats, key=lambda x: -x[3])[:20]:
            print(f"  {mol[:30]:30s} | {host[:25]:25s} | n={n_total} removed={n_out} logKa=[{lk_min:.2f},{lk_max:.2f}]")


if __name__ == "__main__":
    main()
