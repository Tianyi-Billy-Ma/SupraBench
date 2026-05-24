"""
Step 6 — Average duplicate measurements for the same host-guest pair.

"Same pair" is defined by: (molecule, host, ph_bin, solvent_class)
  - ph_bin: pH rounded to nearest 0.5 unit (e.g. pH 6.8–7.2 → bin 7.0)
  - solvent_class: aqueous / unknown
  - Technique is NOT used for grouping (measurements from different methods
    are averaged together as requested).

Within each group:
  - Ka_avg    = geometric mean of ka_25c  (= exp(mean(log_ka_25c)))
  - logKa_avg = arithmetic mean of logka_25c
  - n_measurements = count
  - techniques = unique techniques joined by " | "
  - assay_types = unique assay types joined by " | "

Output: cleaned/06_averaged.csv
"""

import csv
import math
from collections import defaultdict

IN  = "cleaned/05_t_corrected.csv"
OUT = "cleaned/06_averaged.csv"

PH_BIN_SIZE = 0.5   # bin width in pH units


def ph_bin(ph_str: str) -> str:
    """Round pH to nearest PH_BIN_SIZE. Returns '' if not parseable."""
    try:
        ph = float(ph_str)
        binned = round(ph / PH_BIN_SIZE) * PH_BIN_SIZE
        return f"{binned:.1f}"
    except (ValueError, TypeError):
        return ""


def geometric_mean(values):
    n = len(values)
    if n == 0:
        return None
    log_sum = sum(math.log(v) for v in values if v > 0)
    return math.exp(log_sum / n)


def main():
    with open(IN, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Drop rows with no numeric Ka
    rows = [r for r in rows if r.get("ka_25c", "").strip()]
    valid_ka = []
    for r in rows:
        try:
            float(r["ka_25c"])
            valid_ka.append(r)
        except ValueError:
            pass
    rows = valid_ka

    # Build groups
    groups = defaultdict(list)
    for r in rows:
        ph_b = ph_bin(r.get("ph_final", ""))
        key = (
            r.get("molecule", "").strip(),
            r.get("host", "").strip(),
            ph_b,
            r.get("solvent_class", "").strip(),
        )
        groups[key].append(r)

    averaged = []
    for (molecule, host, ph_b, sol_class), group in groups.items():
        ka_vals    = [float(r["ka_25c"]) for r in group]
        logka_vals = []
        for r in group:
            lk = r.get("logka_25c", "").strip()
            if lk:
                try:
                    logka_vals.append(float(lk))
                except ValueError:
                    pass

        ka_geo = geometric_mean(ka_vals)
        lka_mean = sum(logka_vals) / len(logka_vals) if logka_vals else (
            math.log10(ka_geo) if ka_geo else None
        )

        # Collect metadata (representative values from first row + aggregates)
        rep = group[0]
        techniques  = " | ".join(sorted(set(
            r.get("technique", "").strip() for r in group
            if r.get("technique", "").strip()
        )))
        assay_types = " | ".join(sorted(set(
            r.get("assay_type", "").strip() for r in group
            if r.get("assay_type", "").strip()
        )))
        t_corrections = " | ".join(sorted(set(
            r.get("t_correction", "").strip() for r in group
            if r.get("t_correction", "").strip()
        )))
        dois = " | ".join(sorted(set(
            doi.strip()
            for r in group
            for doi in r.get("dois", "").split(" | ")
            if doi.strip()
        )))

        averaged.append({
            "molecule":        molecule,
            "host":            host,
            "ph_bin":          ph_b,
            "solvent_class":   sol_class,
            "ka_avg":          f"{ka_geo:.6g}" if ka_geo else "",
            "logka_avg":       f"{lka_mean:.4f}" if lka_mean is not None else "",
            "n_measurements":  str(len(group)),
            "techniques":      techniques,
            "assay_types":     assay_types,
            "t_corrections":   t_corrections,
            "ph_final":        rep.get("ph_final", ""),
            "ph_assumed":      rep.get("ph_assumed", ""),
            "t_final":         rep.get("t_final", ""),
            "t_assumed":       rep.get("t_assumed", ""),
            "solvent":         rep.get("solvent", ""),
            "indicator":       rep.get("indicator", ""),
            "dois":            dois,
            "interaction_ids": " | ".join(r.get("interaction_id", "") for r in group),
        })

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        all_keys = list(averaged[0].keys()) if averaged else []
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(averaged)

    multi = [r for r in averaged if int(r["n_measurements"]) > 1]
    print(f"Input rows:          {len(rows)}")
    print(f"Unique pairs:        {len(averaged)} → {OUT}")
    print(f"Pairs with >1 meas.: {len(multi)}")
    print(f"Max measurements:    {max(int(r['n_measurements']) for r in averaged)}")


if __name__ == "__main__":
    main()
