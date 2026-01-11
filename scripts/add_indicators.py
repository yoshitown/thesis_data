#!/usr/bin/env python3
"""Invert observed B_score into five Tanaka indicators by per-sample optimization.

Creates additional columns: `H,S,A,B_indicator,C,B_recalc` and writes to output CSV.

This script uses `compute_B_score` from `scripts/b_score_calculator.py` and
solves, for each row, a bounded optimization problem:

    minimize_x (compute_B_score(x) - B_obs)^2 + lambda_reg * ||x - x_prior||^2

where `x_prior` is random (uniform in (0.01,0.99)) as requested.
"""
import argparse
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from scripts.b_score_calculator import compute_B_score


def invert_sample(B_obs: float, x_prior: np.ndarray, lambda_reg: float, maxiter: int) -> Tuple[np.ndarray, float, object]:
    bounds = [(1e-3, 1.0 - 1e-3)] * 5

    def obj(x: np.ndarray) -> float:
        B = compute_B_score(x[0], x[1], x[2], x[3], x[4])
        return float((B - B_obs) ** 2 + lambda_reg * np.sum((x - x_prior) ** 2))

    res = minimize(obj, x0=x_prior, bounds=bounds, method="L-BFGS-B", options={"maxiter": maxiter})
    x_opt = res.x
    B_recalc = float(compute_B_score(x_opt[0], x_opt[1], x_opt[2], x_opt[3], x_opt[4]))
    return x_opt, B_recalc, res


def process(df: pd.DataFrame, lambda_reg: float, maxiter: int, seed: int = None, limit: int = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    out_cols = {"H": [], "S": [], "A": [], "B_indicator": [], "C": [], "B_recalc": []}

    it = df.itertuples(index=False)
    total = len(df) if limit is None else min(len(df), limit)

    for i, row in enumerate(df.itertuples(index=False)):
        if limit is not None and i >= limit:
            break
        B_obs = float(row.B_score)
        x_prior = rng.uniform(0.01, 0.99, size=5)
        x_opt, B_recalc, res = invert_sample(B_obs, x_prior, lambda_reg=lambda_reg, maxiter=maxiter)

        out_cols["H"].append(x_opt[0])
        out_cols["S"].append(x_opt[1])
        out_cols["A"].append(x_opt[2])
        out_cols["B_indicator"].append(x_opt[3])
        out_cols["C"].append(x_opt[4])
        out_cols["B_recalc"].append(B_recalc)

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"Processed {i+1}/{total} rows", file=sys.stderr)

    # attach new cols
    for k, vals in out_cols.items():
        df[k] = vals

    return df


def main(argv=None):
    p = argparse.ArgumentParser(description="Invert B_score into five Tanaka indicators (per-sample optimization).")
    p.add_argument("--input", required=True, help="input CSV file (must contain r,s,B_score,condition)")
    p.add_argument("--output", required=True, help="output CSV path")
    p.add_argument("--lambda_reg", type=float, default=0.01, help="regularization weight")
    p.add_argument("--maxiter", type=int, default=500, help="max iterations for optimizer")
    p.add_argument("--seed", type=int, default=42, help="random seed for x_prior generation")
    p.add_argument("--limit", type=int, default=None, help="optional limit number of rows to process")
    args = p.parse_args(argv)

    df = pd.read_csv(args.input)
    if not {"r", "s", "B_score", "condition"}.issubset(set(df.columns)):
        raise SystemExit("Input CSV must contain columns: r,s,B_score,condition")

    df_out = process(df.copy(), lambda_reg=args.lambda_reg, maxiter=args.maxiter, seed=args.seed, limit=args.limit)

    # report RMSE
    rmse = np.sqrt(np.mean((df_out["B_recalc"].astype(float) - df_out["B_score"].astype(float)) ** 2))
    print(f"RMSE between observed B_score and recomputed B_recalc: {rmse:.6g}")

    df_out.to_csv(args.output, index=False)
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
