"""
Step 2 — Parse numeric fields.
Parses Ka → ka_numeric (float M⁻¹)
         T  → t_numeric  (float °C)
         pH → ph_numeric (float)
         log_ka → logka_numeric (float, recomputed from ka_numeric if possible)
Output: cleaned/02_parsed.csv
"""

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_cleaning.utils import parse_ka, parse_temperature, parse_ph

IN  = "cleaned/01_merged.csv"
OUT = "cleaned/02_parsed.csv"


def main():
    with open(IN, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ok_ka = 0
    ok_t  = 0
    ok_ph = 0

    for r in rows:
        # Ka
        ka_num = parse_ka(r.get("ka", ""))
        if ka_num is None:
            ka_num = parse_ka(r.get("logka", ""))   # fallback: try logka col
        r["ka_numeric"] = str(ka_num) if ka_num is not None else ""

        # log Ka — prefer recomputed from ka_numeric for consistency
        if ka_num and ka_num > 0:
            r["logka_numeric"] = f"{math.log10(ka_num):.4f}"
            ok_ka += 1
        else:
            # fallback to stored log_ka
            lka = parse_ka(r.get("log_ka", ""))   # sometimes stored as plain float
            if lka is None:
                try:
                    lka = float(r.get("log_ka", "").strip())
                except (ValueError, AttributeError):
                    lka = None
            r["logka_numeric"] = f"{lka:.4f}" if lka is not None else ""

        # Temperature
        t = parse_temperature(r.get("t", ""))
        r["t_numeric"] = str(t) if t is not None else ""
        if t is not None:
            ok_t += 1

        # pH
        ph = parse_ph(r.get("ph", ""))
        r["ph_numeric"] = str(ph) if ph is not None else ""
        if ph is not None:
            ok_ph += 1

    # Write
    all_keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(f"Parsed {n} rows → {OUT}")
    print(f"  ka_numeric:    {ok_ka}/{n} ({100*ok_ka/n:.1f}%)")
    print(f"  t_numeric:     {ok_t}/{n}  ({100*ok_t/n:.1f}%)")
    print(f"  ph_numeric:    {ok_ph}/{n}  ({100*ok_ph/n:.1f}%)")
    print(f"  Missing ka:    {n-ok_ka} rows — these will be dropped in later steps")


if __name__ == "__main__":
    main()
