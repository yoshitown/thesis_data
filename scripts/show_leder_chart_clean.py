"""Tanaka test radar chart generator with full multi-run/t0 support.

CSV input supports columns: compound,tR,condition (optional), t0 (optional per-row).
If t0 is missing for a row, the global --t0 is used. Conditions are arbitrary strings.

Usage examples:
  - Write demo multi-run CSV:
      python show_leder_chart_clean.py --write-demo-multi demo_multi_input.csv
  - Run with input CSV and default t0=1.0:
      python show_leder_chart_clean.py -i demo_multi_input.csv -o out.png

"""

from typing import Sequence, Optional, Dict
import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_k(t_R: float, t_0: float) -> float:
    if t_0 <= 0:
        raise ValueError("t0 (dead time) must be > 0")
    return (t_R - t_0) / t_0


def calculate_alpha(k_retained: float, k_least_retained: float) -> float:
    if k_least_retained <= 0:
        return float("nan")
    return k_retained / k_least_retained


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    max_values = df.max(axis=1)
    safe_max = max_values.replace(0, np.nan)
    normalized = df.div(safe_max, axis=0).fillna(0.0)
    return normalized


def draw_radar_chart(
    df: pd.DataFrame,
    labels: Sequence[str],
    title: str = "Tanaka Test Radar Chart",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    num_vars = len(labels)
    if df.shape[0] != num_vars:
        raise ValueError("labels length and df row count must match")

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    # Use black, larger labels per user request
    ax.set_xticklabels(labels, color="black", size=12)

    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    yticks = [f"{v:.1f}" for v in np.arange(0.0, 1.01, 0.2)]
    ax.set_yticklabels(yticks, color="black", size=10)
    ax.set_ylim(0, 1.0)

    ax.set_title(title, y=1.08, color="black", fontsize=16)

    base_colors = [
        "#2ca02c",
        "#1f77b4",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]

    for i, col in enumerate(df.columns):
        values = df[col].tolist()
        values += values[:1]
        color = base_colors[i % len(base_colors)]
        ax.plot(
            angles,
            values,
            color=color,
            linewidth=1.5,
            linestyle="solid",
            label=col,
        )
        ax.fill(angles, values, color=color, alpha=0.25)

    ax.legend(
        loc="lower left",
        bbox_to_anchor=(1.05, 0.1),
        fancybox=True,
        shadow=True,
    )
    # make legend text black and slightly larger
    legend = ax.get_legend()
    if legend is not None:
        for t in legend.get_texts():
            t.set_color("black")
            t.set_fontsize(10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved radar chart to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def draw_empty_radar_chart(
    labels: Sequence[str],
    title: str = "Tanaka Test Radar Chart",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Draw an empty radar chart template (axes, ticks, labels) with larger black text."""
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="black", size=20)

    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    yticks = [f"{v:.1f}" for v in np.arange(0.0, 1.01, 0.2)]
    ax.set_yticklabels(yticks, color="black", size=10)
    ax.set_ylim(0, 1.0)

    ax.set_title(title, y=1.08, color="black", fontsize=20)

    # plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=800)
        print(f"Saved empty radar chart to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def read_compound_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize lower-case column keys for lookup
    cols_lc = {c.lower(): c for c in df.columns}
    if "compound" not in cols_lc or "tr" not in cols_lc:
        raise ValueError("Input CSV must contain columns: compound,tR (and optional condition,t0)")
    # rename to canonical lower names for simpler handling
    rename_map = {cols_lc["compound"]: "compound", cols_lc["tr"]: "tr"}
    if "condition" in cols_lc:
        rename_map[cols_lc["condition"]] = "condition"
    else:
        df["condition"] = "run1"
    if "t0" in cols_lc:
        rename_map[cols_lc["t0"]] = "t0"
    df = df.rename(columns=rename_map)
    # ensure columns exist
    if "condition" not in df.columns:
        df["condition"] = "run1"
    if "t0" not in df.columns:
        df["t0"] = np.nan
    return df


def build_indicator_dataframe(k_by_condition: Dict[str, Dict[str, float]], cond_map: dict) -> pd.DataFrame:
    """Build indicators A-F from k_by_condition using condition mapping.

    cond_map keys: 'abc', 'd', 'e', 'f'
    """
    def k_of(cond: str, compound: str) -> float:
        d = k_by_condition.get(cond, {})
        if compound not in d:
            raise KeyError(f"Compound '{compound}' missing in condition '{cond}'")
        return d[compound]

    abc = cond_map.get("abc", "run1")
    d_cond = cond_map.get("d", "runD")
    e_cond = cond_map.get("e", "runE")
    f_cond = cond_map.get("f", "runF")

    k_pentyl = k_of(abc, "Pentylbenzene")
    alpha_hydrophobicity = calculate_alpha(k_pentyl, k_of(abc, "Butylbenzene"))
    alpha_steric = calculate_alpha(k_of(abc, "Triphenylene"), k_of(abc, "o-Terphenyl"))

    alpha_h_bonding = calculate_alpha(k_of(d_cond, "Caffeine"), k_of(d_cond, "Phenol"))
    alpha_ie_ph7 = calculate_alpha(k_of(e_cond, "Benzylamine"), k_of(e_cond, "Phenol"))
    alpha_ie_ph3 = calculate_alpha(k_of(f_cond, "Benzylamine"), k_of(f_cond, "Phenol"))

    column1 = {
        "A_Retention_k": k_pentyl,
        "B_Hydrophobicity_alpha": alpha_hydrophobicity,
        "C_Steric_alpha": alpha_steric,
        "D_H_Bonding_alpha": alpha_h_bonding,
        "E_IE_pH7_alpha": alpha_ie_ph7,
        "F_IE_pH3_alpha": alpha_ie_ph3,
    }

    column2 = {
        "A_Retention_k": 5.0,
        "B_Hydrophobicity_alpha": 2.3,
        "C_Steric_alpha": 1.8,
        "D_H_Bonding_alpha": 0.60,
        "E_IE_pH7_alpha": 0.35,
        "F_IE_pH3_alpha": 0.01,
    }

    df = pd.DataFrame({"Column 1 (Sample)": column1, "Column 2 (Hypothetical)": column2})
    return df


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Tanaka test radar chart generator (multi-run)")
    p.add_argument("-i", "--input", help="CSV file with columns compound,tR,condition (optional),t0 (optional)")
    p.add_argument("--t0", type=float, default=1.0, help="Default dead time t0 (used when row t0 missing)")
    p.add_argument("-o", "--output", default="tanaka_radar.png", help="Output PNG path")
    p.add_argument("--no-show", action="store_true", help="Do not show plot window (headless)")
    p.add_argument("--write-demo-multi", help="Write demo multi-run CSV to given path and exit")
    p.add_argument("--cond-abc", default="run1", help="Condition name providing A/B/C indicators (default: run1)")
    p.add_argument("--cond-d", default="runD", help="Condition name providing D indicator (default: runD)")
    p.add_argument("--cond-e", default="runE", help="Condition name providing E indicator (default: runE)")
    p.add_argument("--cond-f", default="runF", help="Condition name providing F indicator (default: runF)")
    p.add_argument("--empty-template", action="store_true", help="Generate an empty Tanaka radar chart template (no data) and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # If user requested an empty template, draw it and exit
    if getattr(args, "empty_template", False):
        labels = [
            "A: Retention (k)",
            "B: Hydrophobicity (α)",
            "C: Steric selectivity (α)",
            "D: Hydrogen-bonding (α)",
            "E: Ion-exchange pH>7 (α)",
            "F: Ion-exchange pH<3 (α)",
        ]
        draw_empty_radar_chart(labels, title="Tanaka Test Empty Radar Chart", save_path=args.output, show=(not args.no_show))
        return

    if args.write_demo_multi:
        demo_path = Path(args.write_demo_multi)
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        demo_csv = (
            "compound,tR,condition\n"
            "Uracil,1.0,run1\n"
            "Butylbenzene,2.50,run1\n"
            "Pentylbenzene,4.70,run1\n"
            "o-Terphenyl,3.00,run1\n"
            "Triphenylene,4.90,run1\n"
            "Caffeine,1.50,runD\n"
            "Phenol,1.20,runD\n"
            "Benzylamine,2.30,runE\n"
            "Phenol,1.10,runE\n"
            "Benzylamine,2.10,runF\n"
            "Phenol,1.20,runF\n"
        )
        demo_path.write_text(demo_csv, encoding="utf-8")
        print(f"Wrote demo multi-run CSV to: {demo_path}")
        return

    if not args.input:
        print("No input CSV provided — running built-in demo and saving to", args.output)
        # built-in demo uses default t0
        demo_rows = [
            ("Uracil", args.t0, "run1"),
            ("Butylbenzene", 2.50, "run1"),
            ("Pentylbenzene", 4.70, "run1"),
            ("o-Terphenyl", 3.00, "run1"),
            ("Triphenylene", 4.90, "run1"),
            ("Caffeine", 1.50, "runD"),
            ("Phenol", 1.20, "runD"),
            ("Benzylamine", 2.30, "runE"),
            ("Phenol", 1.10, "runE"),
            ("Benzylamine", 2.10, "runF"),
            ("Phenol", 1.20, "runF"),
        ]
        df_demo = pd.DataFrame(demo_rows, columns=["compound", "tr", "condition"])
        df = df_demo
    else:
        path = Path(args.input)
        if not path.exists():
            print(f"Input file not found: {path}", file=sys.stderr)
            sys.exit(2)
        df = read_compound_csv(path)

    # Compute k per condition using per-row t0 if present else global --t0
    k_by_condition: Dict[str, Dict[str, float]] = defaultdict(dict)
    for _, row in df.iterrows():
        name = str(row["compound"]).strip()
        try:
            tR = float(row["tr"])
        except Exception:
            tR = float("nan")
        cond = str(row.get("condition", "run1"))
        t0_row = row.get("t0", np.nan)
        if not np.isnan(t0_row):
            t0_use = float(t0_row)
        else:
            t0_use = float(args.t0)
        try:
            k_val = calculate_k(tR, t0_use)
        except Exception:
            k_val = float("nan")
        k_by_condition[cond][name] = k_val

    cond_map = {"abc": args.cond_abc, "d": args.cond_d, "e": args.cond_e, "f": args.cond_f}

    df_ind = build_indicator_dataframe(k_by_condition, cond_map)
    normalized = normalize_df(df_ind)
    labels = [
        "A: Retention (k)",
        "B: Hydrophobicity (α)",
        "C: Steric selectivity (α)",
        "D: Hydrogen-bonding (α)",
        "E: Ion-exchange pH>7 (α)",
        "F: Ion-exchange pH<3 (α)",
    ]
    draw_radar_chart(normalized, labels, title="Tanaka Test Normalized Radar Chart", save_path=args.output, show=(not args.no_show))


if __name__ == "__main__":
    main()
