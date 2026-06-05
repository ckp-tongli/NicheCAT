#!/usr/bin/env python3
"""
dim2_stability.py  —  Dimension 2: Cross-sample stability
══════════════════════════════════════════════════════════
How consistent is each niche's cell-type composition across samples?

Metric: Jensen–Shannon Divergence (JSD) of each sample's composition vector
vs. the niche's global centroid (unweighted mean over samples).
  JSD ∈ [0, 1]  (base-2 bits);  lower = more stable / consistent.

Implementation notes:
  · Centroid = unweighted mean over samples (each sample counts equally,
    not biased by sample size).  This is the recommended O(n) approach.
  · All per-sample composition vectors MUST be aligned to the SAME global
    cell-type vocabulary (missing types → 0) before computing JSD.
    Failing to align produces vectors of different lengths and invalid results.
  · Niches present in only 1 sample have no cross-sample stability → N/A.

Requires: composition.csv from validate.py (Branch A or B only).
If absent (Branch C), this script exits with a clear message.

Usage
  python dim2_stability.py --data niche_output/clean_data.h5ad \\
         [--composition niche_output/composition.csv] \\
         [--out-dir niche_output] [--jsd-stable 0.3]

Output
  {out_dir}/dim2_stability.csv
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


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon Divergence in [0,1] (base-2 bits).
    Both inputs are normalised internally; small epsilon avoids log(0)."""
    from scipy.special import rel_entr
    p = np.asarray(p, dtype=float) + 1e-12
    q = np.asarray(q, dtype=float) + 1e-12
    p /= p.sum(); q /= q.sum()
    m  = 0.5 * (p + q)
    ln2 = np.log(2)
    return float(
        0.5 * np.sum(rel_entr(p, m)) / ln2
        + 0.5 * np.sum(rel_entr(q, m)) / ln2
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data",        required=True, help="clean_data.h5ad from validate.py")
    ap.add_argument("--composition", default=None,  help="composition.csv (default: auto-detected alongside --data)")
    ap.add_argument("--out-dir",     default=None,  help="Output directory (default: same folder as --data)")
    ap.add_argument("--jsd-stable",  type=float, default=0.3,
                    help="Median JSD < threshold → 'stable' annotation (default: 0.3)")
    args = ap.parse_args()

    try:
        import anndata as ad
    except ImportError:
        print("[dim2] Installing anndata..."); _install("anndata"); import anndata as ad

    try:
        from scipy.special import rel_entr  # noqa: F401  (imported inside jsd())
    except ImportError:
        _install("scipy")

    data_path = Path(args.data)
    out       = Path(args.out_dir) if args.out_dir else data_path.parent
    out.mkdir(parents=True, exist_ok=True)

    # Auto-locate composition.csv
    comp_path = Path(args.composition) if args.composition else data_path.parent / "composition.csv"
    if not comp_path.exists():
        sys.exit(
            "ERROR: composition.csv not found — dim2 requires cell-type composition data.\n"
            f"  Expected at: {comp_path}\n"
            "  Did validate.py run with --celltype-col (Branch A) or --deconv (Branch B)?\n"
            "  Branch C data cannot run dim2. Only dims 1 and 4 are available for Branch C."
        )

    print("=" * 60)
    print("dim2_stability.py  —  Cross-sample stability (JSD)")
    print("=" * 60)

    adata    = ad.read_h5ad(data_path)
    comp_df  = pd.read_csv(comp_path, index_col=["niche", "sample"])
    cell_types = comp_df.columns.tolist()
    all_niches = sorted(comp_df.index.get_level_values("niche").unique())

    print(f"\n  {len(all_niches)} niches  ×  {len(cell_types)} cell types")
    print(f"  [NOTE] Composition vectors aligned to global vocabulary of {len(cell_types)} types "
          "(missing entries are 0)")
    print(f"  [NOTE] Centroid = unweighted mean over samples (each sample counts equally)")

    records = []
    for niche in all_niches:
        if niche not in comp_df.index.get_level_values("niche"):
            continue

        # All samples that contain this niche
        niche_comp = comp_df.loc[niche]          # shape (n_samples, n_cell_types)
        # reindex to global vocabulary so all vectors have the same length (PITFALL: vocab alignment)
        niche_comp = niche_comp.reindex(columns=cell_types, fill_value=0.0)
        n_s = len(niche_comp)

        if n_s < 2:
            records.append({
                "niche": niche, "n_samples": n_s,
                "jsd_mean": float("nan"), "jsd_median": float("nan"),
                "jsd_max":  float("nan"), "stable": None,
                "note": "only 1 sample — cross-sample stability not computable",
            })
            continue

        # Global centroid: unweighted mean over samples
        centroid  = niche_comp.values.mean(axis=0)
        jsd_vals  = [jsd(row, centroid) for row in niche_comp.values]
        jsd_arr   = np.array(jsd_vals)

        records.append({
            "niche":      niche,
            "n_samples":  n_s,
            "jsd_mean":   round(float(jsd_arr.mean()),   4),
            "jsd_median": round(float(np.median(jsd_arr)), 4),
            "jsd_max":    round(float(jsd_arr.max()),    4),
            "stable":     bool(np.median(jsd_arr) < args.jsd_stable),
            "note":       "",
        })

    result_df = pd.DataFrame(records)
    out_path  = out / "dim2_stability.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}")
    print("\nSummary  (JSD vs centroid; lower = more stable across samples):")
    print(result_df[["niche", "n_samples", "jsd_mean", "jsd_median", "stable", "note"]].to_string(index=False))
    print("\nInterpretation guide:")
    print(f"  Median JSD < {args.jsd_stable}  → niche composition is consistent across samples (stable)")
    print(f"  Median JSD > 0.5   → high variability; check if subgroups drive the instability")
    print("\n=== dim2_stability.py complete ===")


if __name__ == "__main__":
    main()
