#!/usr/bin/env python3
"""Predict tR for raw_data runs from Tanaka indicators.

Reads a preprocessed indicators CSV (rows with columns including
`condition,H,S,A,B_indicator,C`) and a baseline measurements CSV
(rows with `compound,tR,condition`). For each condition in the
indicators file the script computes mean indicators and, using a
user-specified baseline compound measurement, reconstructs k' for
several target compounds via the selection-coefficient definitions
provided by the user mapping.

The script is intentionally conservative: it only computes a compound's
k' if the required baseline or intermediate k' is available; missing
values are skipped with a warning.

Example:
    python -m scripts.predict_tr_from_indicators \
        --indicators preprocessed_data/selected_scores_with_indicators.csv \
        --baseline raw_data/r0.60_s0.38/run_001.csv \
        --baseline-compound Phenol \
        --out predicted_tr.csv

"""
from pathlib import Path
import argparse
import sys
import math
import pandas as pd
from typing import Dict, Any


DEFAULT_MAPPING = {
    # indicator -> (num, den) meaning alpha = k(num) / k(den)
    "H": ("Ethylbenzene", "Toluene"),
    "S": ("1,2,4-TMB", "Ethylbenzene"),
    "A": ("Phenol", "Nitrobenzene"),
    "B": ("Aniline", "Nitrobenzene"),
    "C": ("Benzylamine", "Phenol"),
}

# Typical k' ranges (min, max) for common compounds at 80% ACN, room temp
# These are applied to predicted k' values to keep estimates in plausible ranges.
DEFAULT_K_RANGES = {
    "Toluene": (1.0, 3.0),
    "Ethylbenzene": (2.0, 5.0),
    "Phenol": (1.0, 3.0),
    "Nitrobenzene": (2.0, 4.0),
}


def clip_k_if_needed(k: float, compound: str, clip_enabled: bool) -> float:
    if not clip_enabled:
        return k
    if compound in DEFAULT_K_RANGES:
        lo, hi = DEFAULT_K_RANGES[compound]
        if k < lo:
            print(f"Info: predicted k' for {compound}={k:.3g} below typical min {lo}; clipping to {lo}", file=sys.stderr)
            return float(lo)
        if k > hi:
            print(f"Info: predicted k' for {compound}={k:.3g} above typical max {hi}; clipping to {hi}", file=sys.stderr)
    return float(k)


def read_indicators(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"condition", "H", "S", "A", "B_indicator", "C"}
    if not expected.intersection(set(df.columns)):
        # be tolerant: some files use 'B' instead of 'B_indicator'
        if "B" in df.columns and "B_indicator" not in df.columns:
            df = df.rename(columns={"B": "B_indicator"})
    return df


def read_baseline(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # require compound,tR and optional condition,t0
    return df


def k_from_tr(tR: float, t0: float) -> float:
    return (tR - t0) / t0


def tr_from_k(k: float, t0: float) -> float:
    return t0 * (1.0 + k)


def predict(args: argparse.Namespace) -> pd.DataFrame:
    ind_df = read_indicators(Path(args.indicators))
    base_df = read_baseline(Path(args.baseline)) if args.baseline else None

    # group by condition in indicators
    groups = ind_df.groupby("condition")

    rows = []

    # scales convert 0..1 indicator into alpha; default 1.0 (user can supply)
    scales = {"H": args.scale_H, "S": args.scale_S, "A": args.scale_A, "B": args.scale_B, "C": args.scale_C}

    clip_enabled = not getattr(args, "no_clip", False)
    for cond, g in groups:
        mean = g[["H", "S", "A", "B_indicator", "C"]].mean()
        # canonicalize keys
        mean_dict: Dict[str, float] = {
            "H": float(mean["H"]),
            "S": float(mean["S"]),
            "A": float(mean["A"]),
            "B": float(mean["B_indicator"]),
            "C": float(mean["C"]),
        }

        # find baseline measurement for this condition
        baseline_row = None
        if base_df is not None:
            # try exact condition match first
            cand = base_df[(base_df["compound"] == args.baseline_compound) & (base_df.get("condition", None) == cond)]
            if len(cand) == 0:
                # fallback: any row with baseline compound
                cand = base_df[base_df["compound"] == args.baseline_compound]
            if len(cand) > 0:
                baseline_row = cand.iloc[0]

        if baseline_row is None:
            print(f"Warning: no baseline measurement for compound '{args.baseline_compound}' (condition={cond}); skipping condition", file=sys.stderr)
            continue

        t0_use = float(baseline_row.get("t0", args.t0))
        tR_base = float(baseline_row["tR"])
        k_base = k_from_tr(tR_base, t0_use)

        # store baseline
        rows.append({"condition": cond, "compound": args.baseline_compound, "k": k_base, "tR": tR_base, "note": "baseline"})

        # derive other compounds according to DEFAULT_MAPPING
        # compute alphas from mean indicators using scale
        alphas = {}
        for ind_key in ["H", "S", "A", "B", "C"]:
            alphas[ind_key] = float(mean_dict[ind_key]) * float(scales[ind_key])

        # compute Ethyl/Toluene/1,2,4-TMB chain if baseline is Ethylbenzene
        if args.baseline_compound == "Ethylbenzene":
            # H: alpha_H = k(Ethyl)/k(Toluene) => k(Toluene) = k(Ethyl) / alpha_H
            alpha_H = max(alphas["H"], 1e-8)
            k_toluene = k_base / alpha_H
            k_toluene = clip_k_if_needed(k_toluene, "Toluene", clip_enabled)
            rows.append({"condition": cond, "compound": "Toluene", "k": k_toluene, "tR": tr_from_k(k_toluene, t0_use), "note": "from H"})

            # S: alpha_S = k(1,2,4-TMB)/k(Ethyl) => k(1,2,4-TMB) = alpha_S * k(Ethyl)
            alpha_S = alphas["S"]
            k_124TMB = alpha_S * k_base
            k_124TMB = clip_k_if_needed(k_124TMB, "1,2,4-TMB", clip_enabled)
            rows.append({"condition": cond, "compound": "1,2,4-TMB", "k": k_124TMB, "tR": tr_from_k(k_124TMB, t0_use), "note": "from S"})

        # If baseline is Phenol, derive Nitrobenzene (and Aniline) and Benzylamine
        if args.baseline_compound == "Phenol":
            # k_base is k(Phenol)
            # A: alpha_A = k(Phenol) / k(Nitrobenzene) => k(Nitrobenzene) = k(Phenol) / alpha_A
            alpha_A = max(alphas["A"], 1e-8)
            k_nitro = k_base / alpha_A
            k_nitro = clip_k_if_needed(k_nitro, "Nitrobenzene", clip_enabled)
            rows.append({"condition": cond, "compound": "Nitrobenzene", "k": k_nitro, "tR": tr_from_k(k_nitro, t0_use), "note": "from A"})

            # B: alpha_B = k(Aniline) / k(Nitrobenzene) => k(Aniline) = alpha_B * k_nitro
            alpha_B = alphas["B"]
            k_aniline = alpha_B * k_nitro
            k_aniline = clip_k_if_needed(k_aniline, "Aniline", clip_enabled)
            rows.append({"condition": cond, "compound": "Aniline", "k": k_aniline, "tR": tr_from_k(k_aniline, t0_use), "note": "from B and Nitrobenzene"})

            # C: alpha_C = k(Benzylamine) / k(Phenol) => k(Benzylamine) = alpha_C * k(Phenol)
            alpha_C = alphas["C"]
            k_benzyl = alpha_C * k_base
            k_benzyl = clip_k_if_needed(k_benzyl, "Benzylamine", clip_enabled)
            rows.append({"condition": cond, "compound": "Benzylamine", "k": k_benzyl, "tR": tr_from_k(k_benzyl, t0_use), "note": "from C and Phenol baseline"})

        # A/B chain requires Phenol/Nitrobenzene
        # If baseline file contains Phenol or Nitrobenzene, use them; else skip
        # A: alpha_A = k(Phenol)/k(Nitrobenzene)
        # B: alpha_B = k(Aniline)/k(Nitrobenzene)
        nitro_row = base_df[(base_df["compound"] == "Nitrobenzene") & (base_df.get("condition", None) == cond)] if base_df is not None else pd.DataFrame()
        phenol_row = base_df[(base_df["compound"] == "Phenol") & (base_df.get("condition", None) == cond)] if base_df is not None else pd.DataFrame()
        if len(nitro_row) > 0 and len(phenol_row) > 0:
            t0_n = float(nitro_row.iloc[0].get("t0", args.t0))
            k_nitro = k_from_tr(float(nitro_row.iloc[0]["tR"]), t0_n)
            # from A: k_phenol = alpha_A * k_nitro
            alpha_A = alphas["A"]
            k_phenol = alpha_A * k_nitro
            k_phenol = clip_k_if_needed(k_phenol, "Phenol", clip_enabled)
            rows.append({"condition": cond, "compound": "Phenol(pred)", "k": k_phenol, "tR": tr_from_k(k_phenol, t0_use), "note": "from A and Nitrobenzene"})
            # from B: k_aniline = alpha_B * k_nitro
            alpha_B = alphas["B"]
            k_aniline = alpha_B * k_nitro
            k_aniline = clip_k_if_needed(k_aniline, "Aniline", clip_enabled)
            rows.append({"condition": cond, "compound": "Aniline", "k": k_aniline, "tR": tr_from_k(k_aniline, t0_use), "note": "from B and Nitrobenzene"})
        else:
            # try alternative: if phenol present but nitro missing, derive nitro from phenol/alpha
            if len(phenol_row) > 0:
                t0_p = float(phenol_row.iloc[0].get("t0", args.t0))
                k_phenol_meas = k_from_tr(float(phenol_row.iloc[0]["tR"]), t0_p)
                # alpha_A = k_phenol / k_nitro => k_nitro = k_phenol / alpha_A
                alpha_A = max(alphas["A"], 1e-8)
                k_nitro_est = k_phenol_meas / alpha_A
                k_nitro_est = clip_k_if_needed(k_nitro_est, "Nitrobenzene", clip_enabled)
                rows.append({"condition": cond, "compound": "Nitrobenzene(pred)", "k": k_nitro_est, "tR": tr_from_k(k_nitro_est, t0_use), "note": "inferred from Phenol and A"})
                # then Aniline from alpha_B
                alpha_B = alphas["B"]
                k_aniline = alpha_B * k_nitro_est
                k_aniline = clip_k_if_needed(k_aniline, "Aniline", clip_enabled)
                rows.append({"condition": cond, "compound": "Aniline", "k": k_aniline, "tR": tr_from_k(k_aniline, t0_use), "note": "from B and inferred Nitro"})

        # C chain: alpha_C = k(Benzylamine)/k(Phenol)
        # if Phenol exists in baseline, use measured; else skip
        if base_df is not None and len(phenol_row) > 0:
            t0_p = float(phenol_row.iloc[0].get("t0", args.t0))
            k_phenol_meas = k_from_tr(float(phenol_row.iloc[0]["tR"]), t0_p)
            alpha_C = alphas["C"]
            k_benzyl = alpha_C * k_phenol_meas
            k_benzyl = clip_k_if_needed(k_benzyl, "Benzylamine", clip_enabled)
            rows.append({"condition": cond, "compound": "Benzylamine", "k": k_benzyl, "tR": tr_from_k(k_benzyl, t0_use), "note": "from C and Phenol"})

    out = pd.DataFrame(rows)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Predict tR from Tanaka indicators using measured baseline compound(s)")
    p.add_argument("--indicators", required=True, help="preprocessed indicators CSV")
    p.add_argument("--baseline", required=True, help="baseline measurements CSV (compound,tR,condition optional,t0 optional)")
    p.add_argument("--baseline-compound", default="Phenol", help="name of baseline compound in baseline CSV")
    p.add_argument("--t0", type=float, default=1.0, help="default t0 if not present in baseline CSV")
    p.add_argument("--scale-H", type=float, default=1.0, help="scale factor to convert H indicator to alpha_H")
    p.add_argument("--scale-S", type=float, default=1.0, help="scale factor to convert S indicator to alpha_S")
    p.add_argument("--scale-A", type=float, default=1.0, help="scale factor to convert A indicator to alpha_A")
    p.add_argument("--scale-B", type=float, default=1.0, help="scale factor to convert B indicator to alpha_B")
    p.add_argument("--scale-C", type=float, default=1.0, help="scale factor to convert C indicator to alpha_C")
    p.add_argument("--no-clip", action="store_true", help="disable clipping of predicted k' to typical ranges")
    p.add_argument("--out", default="predicted_tr.csv", help="output CSV path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = predict(args)
    out.to_csv(args.out, index=False)
    print(f"Wrote predictions to: {args.out}")


if __name__ == "__main__":
    main()
