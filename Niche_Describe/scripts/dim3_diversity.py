#!/usr/bin/env python3
"""
dim3_diversity.py  —  Dimension 3: Internal composition diversity
═════════════════════════════════════════════════════════════════
How mixed vs. dominated is each niche internally?

Metrics (computed on the niche's GLOBAL cell-type composition,
         cell-count-weighted mean across samples):
  Simpson D  = 1 − Σpᵢ²   ∈ [0,1];  higher = more diverse (less dominant)
  Shannon H  = −Σpᵢ ln pᵢ ≥ 0;      higher = more diverse (sensitive to rare types)

Also reports: dominant cell type, its proportion, and count of types > 1%.

Requires: composition.csv from validate.py (Branch A or B only).

Usage
  python dim3_diversity.py --data niche_output/clean_data.h5ad \\
         [--composition niche_output/composition.csv] [--out-dir niche_output]

Output
  {out_dir}/dim3_diversity.csv
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


def simpson_d(p: np.ndarray) -> float:
    """Simpson diversity D = 1 - sum(pi²).  Ignores zero entries."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]; total = p.sum()
    if total == 0:
        return float("nan")
    p /= total
    return float(1.0 - np.sum(p ** 2))


def shannon_h(p: np.ndarray) -> float:
    """Shannon entropy H = -sum(pi * ln pi) in nats.  Ignores zero entries."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]; total = p.sum()
    if total == 0:
        return float("nan")
    p /= total
    return float(-np.sum(p * np.log(p)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data",        required=True, help="clean_data.h5ad from validate.py")
    ap.add_argument("--composition", default=None,  help="composition.csv (default: auto-detected alongside --data)")
    ap.add_argument("--out-dir",     default=None,  help="Output directory (default: same folder as --data)")
    args = ap.parse_args()

    try:
        import anndata as ad
    except ImportError:
        print("[dim3] Installing anndata..."); _install("anndata"); import anndata as ad

    data_path = Path(args.data)
    out       = Path(args.out_dir) if args.out_dir else data_path.parent
    out.mkdir(parents=True, exist_ok=True)

    comp_path = Path(args.composition) if args.composition else data_path.parent / "composition.csv"
    if not comp_path.exists():
        sys.exit(
            "ERROR: composition.csv not found — dim3 requires cell-type composition data.\n"
            f"  Expected at: {comp_path}\n"
            "  Did validate.py run with --celltype-col (Branch A) or --deconv (Branch B)?\n"
            "  Branch C data cannot run dim3. Only dims 1 and 4 are available for Branch C."
        )

    print("=" * 60)
    print("dim3_diversity.py  —  Internal composition diversity")
    print("=" * 60)

    adata      = ad.read_h5ad(data_path)
    comp_df    = pd.read_csv(comp_path, index_col=["niche", "sample"])
    cell_types = comp_df.columns.tolist()
    all_niches = sorted(comp_df.index.get_level_values("niche").unique())

    # Cell counts per (niche, sample) for weighted mean
    obs          = adata.obs[["niche", "sample"]].copy()
    group_sizes  = obs.groupby(["niche", "sample"]).size()

    print(f"\n  {len(all_niches)} niches  ×  {len(cell_types)} cell types")
    print("  Global composition = cell-count-weighted mean over samples")

    records = []
    for niche in all_niches:
        if niche not in comp_df.index.get_level_values("niche"):
            continue

        niche_comp  = comp_df.loc[niche].reindex(columns=cell_types, fill_value=0.0)
        # Cell-count-weighted mean: each sample contributes proportionally to its cell count
        if niche in group_sizes.index.get_level_values("niche"):
            sizes = group_sizes.loc[niche]
            # align index between sizes and niche_comp rows
            common_samples = niche_comp.index.intersection(sizes.index)
            if len(common_samples) > 0:
                w     = sizes.loc[common_samples].values.astype(float)
                mat   = niche_comp.loc[common_samples].values
                glob  = np.average(mat, axis=0, weights=w)
            else:
                glob = niche_comp.values.mean(axis=0)
        else:
            glob = niche_comp.values.mean(axis=0)

        # Normalise to sum 1
        glob_sum = glob.sum()
        glob = glob / glob_sum if glob_sum > 0 else glob

        d            = simpson_d(glob)
        h            = shannon_h(glob)
        n_gt1pct     = int((glob > 0.01).sum())
        dominant_idx = int(np.argmax(glob))
        dominant_ct  = cell_types[dominant_idx]
        dominant_prop = float(glob[dominant_idx])

        if np.isnan(d):
            interpretation = "N/A"
        elif d > 0.7:
            interpretation = "high-diversity (well-mixed)"
        elif d > 0.4:
            interpretation = "medium-diversity"
        else:
            interpretation = "low-diversity (one type dominates)"

        records.append({
            "niche":              niche,
            "simpson_d":          round(d, 4) if not np.isnan(d) else float("nan"),
            "shannon_h":          round(h, 4) if not np.isnan(h) else float("nan"),
            "n_celltypes_gt1pct": n_gt1pct,
            "dominant_celltype":  dominant_ct,
            "dominant_prop":      round(dominant_prop, 4),
            "interpretation":     interpretation,
        })

    result_df = pd.DataFrame(records)
    out_path  = out / "dim3_diversity.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}")
    print("\nSummary  (Simpson D: higher = more mixed; Shannon H: higher = more diverse):")
    print(result_df[["niche", "simpson_d", "shannon_h", "dominant_celltype",
                      "dominant_prop", "interpretation"]].to_string(index=False))
    print("\nInterpretation guide:")
    print("  Simpson D > 0.7  → well-mixed niche, no single dominant cell type")
    print("  Simpson D < 0.4  → one or two cell types dominate")
    print("  Shannon H        → especially sensitive to rare types (compare within same dataset)")
    print("\n=== dim3_diversity.py complete ===")


if __name__ == "__main__":
    main()
