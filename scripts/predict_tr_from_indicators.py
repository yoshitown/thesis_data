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
import numpy as np
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


def estimate_baseline_k(
    base_df: pd.DataFrame,
    compound: str,
    cond: Any,
    r_val: Any,
    s_val: Any,
    t0_default: float,
    min_samples: int = 3,
    use_log: bool = True,
    fit_global_if_sparse: bool = False,
) -> Any:
    """Estimate baseline k' for given compound, condition and (r,s).

    Returns tuple (k_est, t0_used, source) or (None, None, None) if not available.
    """
    if base_df is None or len(base_df) == 0:
        return (None, None, None)

    # select rows for the compound and condition
    rows = base_df[base_df["compound"] == compound]
    if "condition" in base_df.columns:
        rows_cond = rows[rows.get("condition", None) == cond]
    else:
        rows_cond = rows

    # try exact r/s match when available
    exact = None
    if r_val is not None and s_val is not None and "r" in rows_cond.columns and "s" in rows_cond.columns:
        try:
            cand = rows_cond.copy()
            cand_r = cand["r"].astype(float)
            cand_s = cand["s"].astype(float)
            mask = np.isclose(cand_r.values, float(r_val)) & np.isclose(cand_s.values, float(s_val))
            cand_exact = cand[mask]
            if len(cand_exact) > 0:
                exact = cand_exact.iloc[0]
        except Exception:
            exact = None

    if exact is not None:
        t0_use = float(exact.get("t0", t0_default))
        k_val = k_from_tr(float(exact["tR"]), t0_use)
        return (float(k_val), float(t0_use), "measured_exact")

    # try fitting using rows from same condition
    def try_fit(df_rows: pd.DataFrame):
        # build arrays
        ys = []
        Rs = []
        Ss = []
        t0s = []
        for _, r in df_rows.iterrows():
            try:
                t0_i = float(r.get("t0", t0_default))
                k_i = k_from_tr(float(r["tR"]), t0_i)
            except Exception:
                continue
            if not np.isfinite(k_i) or k_i <= 0:
                continue
            ys.append(np.log(k_i) if use_log else k_i)
            Rs.append(float(r.get("r", np.nan)))
            Ss.append(float(r.get("s", np.nan)))
            t0s.append(t0_i)

        if len(ys) < min_samples:
            return (None, None)

        X = np.column_stack([np.ones(len(Rs)), np.array(Rs), np.array(Ss)])
        y = np.array(ys)
        try:
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            x_pred = np.array([1.0, float(r_val) if r_val is not None else 0.0, float(s_val) if s_val is not None else 0.0])
            y_pred = float(x_pred.dot(coeffs))
            k_pred = float(np.exp(y_pred)) if use_log else float(y_pred)
            t0_use = float(np.mean(t0s)) if len(t0s) > 0 else t0_default
            return (k_pred, t0_use)
        except Exception:
            return (None, None)

    # attempt per-condition fit
    if len(rows_cond) >= min_samples:
        k_fit, t0_fit = try_fit(rows_cond)
        if k_fit is not None:
            return (k_fit, t0_fit, "fitted_condition")

    # optional: try global fit across all conditions for this compound
    if fit_global_if_sparse and len(rows) >= min_samples:
        k_fit, t0_fit = try_fit(rows)
        if k_fit is not None:
            return (k_fit, t0_fit, "fitted_global")

    # fallback: use any measured row for compound (first one)
    if len(rows) > 0:
        row0 = rows.iloc[0]
        t0_use = float(row0.get("t0", t0_default))
        try:
            k_val = k_from_tr(float(row0["tR"]), t0_use)
            return (float(k_val), float(t0_use), "measured_fallback")
        except Exception:
            return (None, None, None)

    return (None, None, None)


def predict(args: argparse.Namespace) -> pd.DataFrame:
    ind_df = read_indicators(Path(args.indicators))
    base_df = read_baseline(Path(args.baseline)) if args.baseline else None

    # choose grouping: prefer condition+r+s if present
    if {"r", "s"}.issubset(set(ind_df.columns)):
        group_keys = ["condition", "r", "s"]
    else:
        group_keys = ["condition"]
    groups = ind_df.groupby(group_keys)

    rows = []

    # scales convert 0..1 indicator into alpha; default 1.0 (user can supply)
    scales = {"H": args.scale_H, "S": args.scale_S, "A": args.scale_A, "B": args.scale_B, "C": args.scale_C}

    clip_enabled = not getattr(args, "no_clip", False)
    for group_key, g in groups:
        if len(group_keys) == 3:
            cond, r_val, s_val = group_key
        else:
            cond = group_key
            r_val = None
            s_val = None

        mean = g[["H", "S", "A", "B_indicator", "C"]].mean()
        mean_dict: Dict[str, float] = {
            "H": float(mean["H"]),
            "S": float(mean["S"]),
            "A": float(mean["A"]),
            "B": float(mean["B_indicator"]),
            "C": float(mean["C"]),
        }

        # estimate or find baseline k for this condition,r,s
        k_base = None
        t0_use = float(args.t0)
        note_base = ""
        if base_df is not None:
            k_est, t0_est, src = estimate_baseline_k(
                base_df,
                args.baseline_compound,
                cond,
                r_val,
                s_val,
                args.t0,
                min_samples=args.fit_min_samples,
                use_log=not getattr(args, "no_log_fit", False),
                fit_global_if_sparse=getattr(args, "fit_global_if_sparse", False),
            )
            if k_est is not None:
                k_base = float(k_est)
                t0_use = float(t0_est)
                note_base = f"baseline({src})"

        if k_base is None:
            print(f"Warning: no baseline measurement for compound '{args.baseline_compound}' (condition={cond}, r={r_val}, s={s_val}); skipping", file=sys.stderr)
            continue

        tR_base = tr_from_k(k_base, t0_use)
        rows.append({"condition": cond, "r": r_val, "s": s_val, "compound": args.baseline_compound, "k": k_base, "tR": tR_base, "note": note_base})

        # derive other compounds according to DEFAULT_MAPPING
        # compute alphas from mean indicators using scale
        alphas = {}
        for ind_key in ["H", "S", "A", "B", "C"]:
            alphas[ind_key] = float(mean_dict[ind_key]) * float(scales[ind_key])

        # compute Ethyl/Toluene/1,2,4-TMB chain if baseline is Ethylbenzene
        if args.baseline_compound == "Ethylbenzene":
            alpha_H = max(alphas["H"], 1e-8)
            k_toluene = k_base / alpha_H
            k_toluene = clip_k_if_needed(k_toluene, "Toluene", clip_enabled)
            rows.append({"condition": cond, "r": r_val, "s": s_val, "compound": "Toluene", "k": k_toluene, "tR": tr_from_k(k_toluene, t0_use), "note": "from H"})

            alpha_S = alphas["S"]
            k_124TMB = alpha_S * k_base
            k_124TMB = clip_k_if_needed(k_124TMB, "1,2,4-TMB", clip_enabled)
            rows.append({"condition": cond, "r": r_val, "s": s_val, "compound": "1,2,4-TMB", "k": k_124TMB, "tR": tr_from_k(k_124TMB, t0_use), "note": "from S"})

        # If baseline is Phenol, derive Nitrobenzene (and Aniline) and Benzylamine
        if args.baseline_compound == "Phenol":
            alpha_A = max(alphas["A"], 1e-8)
            k_nitro = k_base / alpha_A
            k_nitro = clip_k_if_needed(k_nitro, "Nitrobenzene", clip_enabled)
            rows.append({"condition": cond, "r": r_val, "s": s_val, "compound": "Nitrobenzene", "k": k_nitro, "tR": tr_from_k(k_nitro, t0_use), "note": "from A"})

            alpha_B = alphas["B"]
            k_aniline = alpha_B * k_nitro
            k_aniline = clip_k_if_needed(k_aniline, "Aniline", clip_enabled)
            rows.append({"condition": cond, "r": r_val, "s": s_val, "compound": "Aniline", "k": k_aniline, "tR": tr_from_k(k_aniline, t0_use), "note": "from B and Nitrobenzene"})

            alpha_C = alphas["C"]
            k_benzyl = alpha_C * k_base
            k_benzyl = clip_k_if_needed(k_benzyl, "Benzylamine", clip_enabled)
            rows.append({"condition": cond, "r": r_val, "s": s_val, "compound": "Benzylamine", "k": k_benzyl, "tR": tr_from_k(k_benzyl, t0_use), "note": "from C and Phenol baseline"})

        # A/B chain: where Nitrobenzene or Phenol measured in baseline file for same condition, try to use them
        nitro_row = base_df[(base_df["compound"] == "Nitrobenzene") & (base_df.get("condition", None) == cond)] if base_df is not None else pd.DataFrame()
        phenol_row = base_df[(base_df["compound"] == "Phenol") & (base_df.get("condition", None) == cond)] if base_df is not None else pd.DataFrame()
        if len(nitro_row) > 0 and len(phenol_row) > 0:
            t0_n = float(nitro_row.iloc[0].get("t0", args.t0))
            k_nitro_meas = k_from_tr(float(nitro_row.iloc[0]["tR"]), t0_n)
            alpha_A = alphas["A"]
            k_phenol_pred = alpha_A * k_nitro_meas
            k_phenol_pred = clip_k_if_needed(k_phenol_pred, "Phenol", clip_enabled)
            rows.append({"condition": cond, "r": r_val, "s": s_val, "compound": "Phenol(pred)", "k": k_phenol_pred, "tR": tr_from_k(k_phenol_pred, t0_use), "note": "from A and Nitrobenzene"})

            alpha_B = alphas["B"]
            k_aniline = alpha_B * k_nitro_meas
            k_aniline = clip_k_if_needed(k_aniline, "Aniline", clip_enabled)
            rows.append({"condition": cond, "r": r_val, "s": s_val, "compound": "Aniline", "k": k_aniline, "tR": tr_from_k(k_aniline, t0_use), "note": "from B and Nitrobenzene"})
        else:
            if len(phenol_row) > 0:
                t0_p = float(phenol_row.iloc[0].get("t0", args.t0))
                k_phenol_meas = k_from_tr(float(phenol_row.iloc[0]["tR"]), t0_p)
                alpha_A = max(alphas["A"], 1e-8)
                k_nitro_est = k_phenol_meas / alpha_A
                k_nitro_est = clip_k_if_needed(k_nitro_est, "Nitrobenzene", clip_enabled)
                rows.append({"condition": cond, "r": r_val, "s": s_val, "compound": "Nitrobenzene(pred)", "k": k_nitro_est, "tR": tr_from_k(k_nitro_est, t0_use), "note": "inferred from Phenol and A"})
                alpha_B = alphas["B"]
                k_aniline = alpha_B * k_nitro_est
                k_aniline = clip_k_if_needed(k_aniline, "Aniline", clip_enabled)
                rows.append({"condition": cond, "r": r_val, "s": s_val, "compound": "Aniline", "k": k_aniline, "tR": tr_from_k(k_aniline, t0_use), "note": "from B and inferred Nitro"})

        # C chain: if Phenol exists in baseline for this condition, derive Benzylamine
        if base_df is not None and len(phenol_row) > 0:
            t0_p = float(phenol_row.iloc[0].get("t0", args.t0))
            k_phenol_meas = k_from_tr(float(phenol_row.iloc[0]["tR"]), t0_p)
            alpha_C = alphas["C"]
            k_benzyl = alpha_C * k_phenol_meas
            k_benzyl = clip_k_if_needed(k_benzyl, "Benzylamine", clip_enabled)
            rows.append({"condition": cond, "r": r_val, "s": s_val, "compound": "Benzylamine", "k": k_benzyl, "tR": tr_from_k(k_benzyl, t0_use), "note": "from C and Phenol"})

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
    p.add_argument("--fit-min-samples", type=int, default=3, help="minimum samples required to fit r/s model per condition")
    p.add_argument("--no-log-fit", action="store_true", help="disable log-space fit for k' (fit k directly)")
    p.add_argument("--fit-global-if-sparse", action="store_true", help="allow global-fit across conditions when per-condition data sparse")
    p.add_argument("--out", default="predicted_tr.csv", help="output CSV path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = predict(args)
    out.to_csv(args.out, index=False)
    print(f"Wrote predictions to: {args.out}")


if __name__ == "__main__":
    main()
