#!/usr/bin/env python3
"""
Add 5 to the `tR` column of a predicted_tr.csv and save result.
Usage:
  python results/r0.66_s0.27/plus5.py --in INPUT_CSV --out OUTPUT_CSV
If no args provided, operates on the file next to this script:
  predicted_tr.csv -> predicted_tr_plus5.csv
"""
import argparse
from pathlib import Path

def main():
    p = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Add 5 to tR column")
    parser.add_argument("--in", dest="input", default=str(p / "predicted_tr.csv"))
    parser.add_argument("--out", dest="output", default=str(p / "predicted_tr_plus5.csv"))
    args = parser.parse_args()

    try:
        import pandas as pd
        df = pd.read_csv(args.input)
        if "tR" not in df.columns:
            raise SystemExit("Input CSV has no 'tR' column")
        df["tR"] = df["tR"] + 5
        df.to_csv(args.output, index=False)
        print(f"Saved: {args.output}")
    except Exception as e:
        # fallback to csv module for simple processing
        import csv
        inp = Path(args.input)
        outp = Path(args.output)
        if not inp.exists():
            raise SystemExit(f"Input file not found: {inp}")
        with inp.open(newline='', encoding='utf-8') as rf, outp.open("w", newline='', encoding='utf-8') as wf:
            reader = csv.DictReader(rf)
            fieldnames = reader.fieldnames
            if fieldnames is None or "tR" not in fieldnames:
                raise SystemExit("Input CSV has no 'tR' column")
            writer = csv.DictWriter(wf, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                try:
                    row["tR"] = str(float(row["tR"]) + 5)
                except Exception:
                    pass
                writer.writerow(row)
        print(f"Saved (csv fallback): {outp}")

if __name__ == "__main__":
    main()
