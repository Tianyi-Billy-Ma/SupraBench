"""
Run the full data cleaning pipeline.

Steps:
  1. Merge raw CSV files
  2. Parse Ka, T, pH to numeric
  3. Filter organic solvents
  4. Fill missing T/pH with standard defaults
  5. Van't Hoff temperature correction to 25 °C
  6. (optional) Remove outliers before averaging
  7. Average duplicate measurements per pair
  8. Re-average after outlier removal (clean final output)

Final outputs in data_cleaning/cleaned/:
  06_averaged.csv          — averaged pairs (with outliers included)
  final_clean.csv          — averaged pairs after outlier removal (recommended)
"""

import os
import sys

# Run from the data_cleaning directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs("cleaned", exist_ok=True)


def run_step(name, module_path):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print('='*60)
    import importlib.util
    spec = importlib.util.spec_from_file_location("step", module_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def main():
    base = os.path.dirname(os.path.abspath(__file__))

    run_step("Step 1: Merge raw datasets",          f"{base}/step1_merge.py")
    run_step("Step 2: Parse Ka / T / pH",           f"{base}/step2_parse_numerics.py")
    run_step("Step 3: Filter organic solvents",     f"{base}/step3_filter_solvents.py")
    run_step("Step 4: Fill missing T / pH",         f"{base}/step4_fill_defaults.py")
    run_step("Step 5: Van't Hoff T correction",     f"{base}/step5_vanthoff.py")

    # Step 6a: Average including outliers (reference)
    run_step("Step 6a: Average pairs (all data)",   f"{base}/step6_average_pairs.py")

    # Step 7: Detect and remove outliers from pre-averaged data
    run_step("Step 7: Outlier removal",             f"{base}/step7_outliers.py")

    # Step 6b: Re-average after outlier removal → final clean file
    print(f"\n{'='*60}")
    print(f"  Step 6b: Re-average after outlier removal")
    print('='*60)

    import importlib.util, shutil
    # Temporarily redirect input of step6 to outlier-cleaned data
    spec = importlib.util.spec_from_file_location("step6b", f"{base}/step6_average_pairs.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Patch the IN path and output path
    mod.IN  = "cleaned/07_no_outliers.csv"
    mod.OUT = "cleaned/final_clean.csv"
    mod.main()

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print('='*60)
    print()
    print("Key outputs:")
    print("  cleaned/final_clean.csv     ← RECOMMENDED: averaged, outliers removed")
    print("  cleaned/06_averaged.csv     ← averaged (outliers NOT removed)")
    print("  cleaned/07_outliers.csv     ← removed outlier rows (for inspection)")
    print("  cleaned/03_dropped_organic.csv ← organic solvent rows (discarded)")
    print()

    # Summary stats
    import csv
    try:
        with open("cleaned/final_clean.csv") as f:
            rows = list(csv.DictReader(f))
        hosts = set(r["host"] for r in rows)
        molecules = set(r["molecule"] for r in rows)
        multi = [r for r in rows if int(r.get("n_measurements","1")) > 1]
        print(f"Final dataset summary:")
        print(f"  Unique pairs:     {len(rows)}")
        print(f"  Unique hosts:     {len(hosts)}")
        print(f"  Unique molecules: {len(molecules)}")
        print(f"  Pairs with >1 measurement averaged: {len(multi)}")
        logka_vals = []
        for r in rows:
            try: logka_vals.append(float(r["logka_avg"]))
            except: pass
        if logka_vals:
            import statistics
            print(f"  logKa range:      {min(logka_vals):.2f} – {max(logka_vals):.2f}")
            print(f"  logKa median:     {statistics.median(logka_vals):.2f}")
    except Exception as e:
        print(f"Could not summarize final output: {e}")


if __name__ == "__main__":
    main()
