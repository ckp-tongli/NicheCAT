#!/usr/bin/env python3
"""
dim1_prevalence.py  —  Dimension 1: Cross-sample prevalence
═════════════════════════════════════════════════════════════
How broadly does each niche appear across samples?

Metrics (all computed on NORMALIZED per-sample proportions, not raw counts):
  SPR   Sample Prevalence Rate = #samples containing niche / #total samples
  Gini  Gini coefficient of per-sample proportions  (0=even, 1=concentrated)
  Conc  Top-X% concentration = % of niche cells from the top-X% richest samples

WARNING: Gini and Conc MUST be computed on normalized proportions
  (niche_cells_in_sample / total_cells_in_sample), NOT raw counts.
  Raw counts give large-sample bias: a niche present equally in every sample
  would look falsely concentrated if sample sizes differ.

Usage
  python dim1_prevalence.py --data niche_output/clean_data.h5ad \\
         [--out-dir niche_output] [--spr-min 0.2] [--top-pct 0.05]

Output
  {out_dir}/dim1_prevalence.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd


def _install(pkg):
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "-q"]
    )


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array.  Returns NaN for all-zero input."""
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum() / (n * v.sum())) - (n + 1) / n)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data",           required=True,  help="clean_data.h5ad from validate.py")
    ap.add_argument("--out-dir",        default=None,   help="Output directory (default: same folder as --data)")
    ap.add_argument("--spr-min",        type=float, default=0.2,  help="SPR ≥ threshold → 'widespread' (default: 0.2)")
    ap.add_argument("--gini-threshold", type=float, default=0.5,  help="Gini < threshold → 'even' (default: 0.5)")
    ap.add_argument("--top-pct",        type=float, default=0.05, help="Top-X%% of samples for concentration (default: 0.05)")
    ap.add_argument("--conc-threshold", type=float, default=0.30, help="Conc < threshold → 'even' (default: 0.30)")
    args = ap.parse_args()

    try:
        import anndata as ad
    except ImportError:
        print("[dim1] Installing anndata..."); _install("anndata"); import anndata as ad

    data_path = Path(args.data)
    out = Path(args.out_dir) if args.out_dir else data_path.parent
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("dim1_prevalence.py  —  Cross-sample prevalence")
    print("=" * 60)

    adata = ad.read_h5ad(data_path)
    obs   = adata.obs[["niche", "sample"]].copy()

    all_niches    = sorted(obs["niche"].unique())
    all_samples   = sorted(obs["sample"].unique())
    total_samples = len(all_samples)
    sample_totals = obs.groupby("sample").size()  # total cells per sample

    print(f"\n  {len(all_niches)} niches × {total_samples} samples")
    print(f"  [NOTE] Gini and Top-{int(args.top_pct*100)}% computed on normalized proportions (not raw counts)")

    records = []
    for niche in all_niches:
        niche_obs = obs[obs["niche"] == niche]

        # Per-sample normalized proportions  (PITFALL 5: normalize before computing Gini)
        per_sample_prop = {
            s: float(niche_obs[niche_obs["sample"] == s].shape[0]) / float(sample_totals[s])
            for s in all_samples
        }
        prop_values = np.array([per_sample_prop[s] for s in all_samples])

        # SPR
        spr = float((prop_values > 0).sum()) / total_samples

        # Gini on normalized proportions
        g = gini(prop_values)

        # Top-X% concentration
        n_top     = max(1, int(np.ceil(total_samples * args.top_pct)))
        n_niche   = niche_obs.shape[0]
        if n_niche > 0:
            top_samples = sorted(all_samples, key=lambda s: per_sample_prop[s], reverse=True)[:n_top]
            top_cells   = niche_obs[niche_obs["sample"].isin(top_samples)].shape[0]
            conc        = float(top_cells) / float(n_niche)
        else:
            conc = float("nan")

        records.append({
            "niche":       niche,
            "n_cells":     int(n_niche),
            "n_samples":   int((prop_values > 0).sum()),
            "spr":         round(spr, 4),
            "gini":        round(g, 4) if not np.isnan(g) else float("nan"),
            f"top{int(args.top_pct*100)}pct_conc": round(conc, 4) if not np.isnan(conc) else float("nan"),
            "widespread":  spr   >= args.spr_min,
            "gini_even":   (g    <  args.gini_threshold) if not np.isnan(g)    else None,
            "conc_even":   (conc <  args.conc_threshold) if not np.isnan(conc) else None,
        })

    result_df = pd.DataFrame(records)
    out_path  = out / "dim1_prevalence.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}")
    print("\nSummary:")
    display_cols = ["niche", "spr", "gini", f"top{int(args.top_pct*100)}pct_conc", "widespread"]
    print(result_df[display_cols].to_string(index=False))
    print("\nInterpretation guide:")
    print(f"  SPR ≥ {args.spr_min}         → niche is widespread across samples")
    print(f"  Gini < {args.gini_threshold}        → niche abundance is evenly distributed across samples")
    print(f"  Top-{int(args.top_pct*100)}% conc < {args.conc_threshold}  → niche is NOT dominated by a handful of samples")
    print("\n=== dim1_prevalence.py complete ===")


if __name__ == "__main__":
    main()
