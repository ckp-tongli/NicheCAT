#!/usr/bin/env python3
"""
dim4_spatial_clustering.py  —  Dimension 4: Single-niche spatial clustering
═════════════════════════════════════════════════════════════════════════════
Is each niche spatially clustered within tissue, or randomly scattered?

Method: binarized Join Count statistic with permutation test + BH/FDR correction.

Algorithm per niche per sample
  1. Binarize: cells of target niche = 1, all other cells = 0.
  2. Build per-sample spatial graph (kNN or radius).
  3. Count observed 1–1 (BB) joins in the graph.
  4. Permute binary labels n_perm times; count BB joins each time.
  5. One-sided p-value = (# perms with BB ≥ observed + 1) / (n_perm + 1).
  6. Aggregate per-niche: median z-score, # samples significant after BH correction.

RULE 1: graphs are ALWAYS built per-sample independently.
        Never pool cells from multiple samples into one graph.
RULE 3: niche labels are CATEGORICAL. They are binarized here.
        Do NOT feed raw niche IDs to Moran's I / Geary's C / Getis-Ord G.

No external spatial-stats libraries required (only scipy + numpy + statsmodels).

Usage
  python dim4_spatial_clustering.py --data niche_output/clean_data.h5ad \\
         [--graph-method knn|radius] [--knn-k 6] [--radius 100] \\
         [--n-perm 999] [--fdr-alpha 0.05] [--seed 42] [--out-dir niche_output]

Outputs
  {out_dir}/dim4_spatial_clustering.csv          — per-niche summary
  {out_dir}/dim4_spatial_clustering_detail.csv   — per-niche × per-sample raw results
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree


def _install(pkg):
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "-q"]
    )


# ── Graph builders (scipy only, no libpysal / esda needed) ────────────────────

def _knn_graph(coords: np.ndarray, k: int) -> sp.csr_matrix:
    """Symmetric, binary kNN adjacency matrix."""
    n = len(coords)
    k = min(k, n - 1)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)      # +1: first neighbour is self
    rows, cols = [], []
    for i, neighbours in enumerate(idx[:, 1:]):  # skip self (col 0)
        rows.extend([i] * len(neighbours) + list(neighbours))
        cols.extend(list(neighbours) + [i] * len(neighbours))
    data = np.ones(len(rows), dtype=np.float32)
    adj  = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    adj.data[:] = 1                           # binarize duplicate edges
    return adj


def _radius_graph(coords: np.ndarray, radius: float) -> sp.csr_matrix:
    """Radius-based binary adjacency matrix."""
    n    = len(coords)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(len(rows), dtype=np.float32)
    adj  = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    adj.data[:] = 1
    return adj


# ── Join Count test ────────────────────────────────────────────────────────────

def _count_bb(z: np.ndarray, adj: sp.csr_matrix) -> int:
    """Count 1–1 (BB) joins in the upper triangle of the adjacency matrix."""
    rows, cols = adj.nonzero()
    upper = rows < cols                       # avoid double-counting
    return int(((z[rows[upper]] == 1) & (z[cols[upper]] == 1)).sum())


def _join_count_test(
    z: np.ndarray,
    adj: sp.csr_matrix,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple:
    """
    Permutation-based Join Count test for spatial clustering of binary z.
    Returns (obs_bb, mean_perm_bb, z_score, p_val_one_sided).
    p_val uses continuity correction: (# perms ≥ obs + 1) / (n_perm + 1).
    """
    z    = np.asarray(z, dtype=int)
    obs  = _count_bb(z, adj)
    perm = np.array([_count_bb(rng.permutation(z), adj) for _ in range(n_perm)])
    mu   = float(perm.mean())
    sigma = float(perm.std()) + 1e-10
    z_score = (obs - mu) / sigma
    p_val   = (float((perm >= obs).sum()) + 1.0) / (n_perm + 1.0)
    return obs, mu, round(z_score, 4), round(p_val, 6)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data",         required=True,  help="clean_data.h5ad from validate.py")
    ap.add_argument("--graph-method", default="knn",  choices=["knn", "radius"],
                    help="Spatial graph type: knn or radius (default: knn)")
    ap.add_argument("--knn-k",  type=int,   default=6,     help="k for kNN graph (default: 6)")
    ap.add_argument("--radius", type=float, default=100.0,
                    help="Radius for radius graph in coordinate units (default: 100). "
                         "Confirm unit with validate.py / references/platform_notes.md")
    ap.add_argument("--n-perm",    type=int,   default=999,  help="Permutations per test (default: 999)")
    ap.add_argument("--fdr-alpha", type=float, default=0.05, help="BH-FDR significance threshold (default: 0.05)")
    ap.add_argument("--seed",      type=int,   default=None,
                    help="Random seed (default: inherits seed from run_config.json)")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: same folder as --data)")
    args = ap.parse_args()

    try:
        import anndata as ad
    except ImportError:
        print("[dim4] Installing anndata..."); _install("anndata"); import anndata as ad
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        print("[dim4] Installing statsmodels..."); _install("statsmodels")
        from statsmodels.stats.multitest import multipletests

    data_path = Path(args.data)
    out = Path(args.out_dir) if args.out_dir else data_path.parent
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("dim4_spatial_clustering.py  —  Single-niche spatial clustering")
    print("=" * 60)

    adata = ad.read_h5ad(data_path)
    obs   = adata.obs[["niche", "sample"]].copy()
    coords_all = adata.obsm["spatial"]        # shape (n_cells, 2)

    # Seed: prefer explicit --seed, then fall back to run_config
    seed = args.seed
    if seed is None:
        seed = adata.uns.get("run_config", {}).get("seed", 42)
    rng = np.random.default_rng(seed)

    all_niches  = sorted(obs["niche"].unique())
    all_samples = sorted(obs["sample"].unique())

    print(f"\n  Graph:       {args.graph_method}  "
          f"(k={args.knn_k})" if args.graph_method == "knn" else
          f"\n  Graph:       {args.graph_method}  (radius={args.radius})")
    print(f"  Permutations: {args.n_perm}  |  FDR α: {args.fdr_alpha}  |  Seed: {seed}")
    print(f"  {len(all_niches)} niches  ×  {len(all_samples)} samples")
    print("\n  RULE 1 in effect: one graph per sample, never pooled across samples.")
    print("  RULE 3 in effect: niche labels binarized (target=1, other=0); "
          "NOT fed as continuous values.\n")

    # ── Per-niche × per-sample tests ──────────────────────────────────────────
    raw_rows = []

    for sample in all_samples:
        s_mask = (obs["sample"] == sample).values
        s_idx  = np.where(s_mask)[0]
        if len(s_idx) < 5:
            print(f"  SKIP sample '{sample}': only {len(s_idx)} cells (< 5).")
            continue

        s_coords  = coords_all[s_idx]
        s_niches  = obs["niche"].values[s_idx]

        # Build graph ONCE per sample (shared across all niches in this sample)
        if args.graph_method == "knn":
            adj = _knn_graph(s_coords, args.knn_k)
        else:
            adj = _radius_graph(s_coords, args.radius)

        if adj.nnz == 0:
            print(f"  WARNING: empty graph for sample '{sample}'. "
                  "Try increasing --knn-k or --radius, or check coordinate units.")
            continue

        for niche in all_niches:
            z          = (s_niches == niche).astype(int)
            n_in_sample = int(z.sum())
            n_total     = len(z)

            if n_in_sample == 0:
                continue                      # niche absent in this sample

            if n_in_sample == n_total:        # entire sample is this niche
                raw_rows.append({
                    "niche": niche, "sample": sample,
                    "n_niche": n_in_sample, "n_total": n_total,
                    "obs_bb": int(adj.nnz // 2), "exp_bb": float("nan"),
                    "z_score": float("nan"),    "p_raw": float("nan"),
                    "note": "all cells are this niche — test inapplicable",
                })
                continue

            obs_bb, exp_bb, z_score, p_raw = _join_count_test(z, adj, args.n_perm, rng)
            raw_rows.append({
                "niche": niche, "sample": sample,
                "n_niche": n_in_sample, "n_total": n_total,
                "obs_bb":  obs_bb,
                "exp_bb":  round(exp_bb, 2),
                "z_score": z_score,
                "p_raw":   p_raw,
                "note":    "",
            })

    if not raw_rows:
        sys.exit(
            "ERROR: no valid niche×sample tests were run.\n"
            "  Check --min-cells in validate.py and --knn-k / --radius here."
        )

    raw_df = pd.DataFrame(raw_rows)

    # ── BH/FDR correction across ALL (niche × sample) tests  (PITFALL 10) ────
    testable = raw_df["p_raw"].notna()
    if testable.sum() > 0:
        _, p_bh, _, _ = multipletests(
            raw_df.loc[testable, "p_raw"].values, alpha=args.fdr_alpha, method="fdr_bh"
        )
        raw_df.loc[testable, "p_bh"] = np.round(p_bh, 6)
    raw_df.loc[~testable, "p_bh"] = float("nan")

    # ── Per-niche summary ─────────────────────────────────────────────────────
    summary_rows = []
    for niche in all_niches:
        sub = raw_df[raw_df["niche"] == niche]
        ok  = sub[sub["p_raw"].notna()]        # testable rows only
        n_tested = len(ok)
        n_sig    = int((ok["p_bh"] < args.fdr_alpha).sum()) if n_tested > 0 else 0

        summary_rows.append({
            "niche":                 niche,
            "n_samples_tested":      n_tested,
            "n_samples_clustered":   n_sig,
            "pct_samples_clustered": round(n_sig / n_tested, 3) if n_tested > 0 else float("nan"),
            "median_z_score":        round(float(ok["z_score"].median()), 3) if n_tested > 0 else float("nan"),
            "median_obs_bb":         round(float(ok["obs_bb"].median()),  1) if n_tested > 0 else float("nan"),
            "median_exp_bb":         round(float(ok["exp_bb"].median()),  1) if n_tested > 0 else float("nan"),
            "spatially_clustered":   n_sig > 0,
        })

    summary_df = pd.DataFrame(summary_rows)

    # ── Write outputs ─────────────────────────────────────────────────────────
    detail_path  = out / "dim4_spatial_clustering_detail.csv"
    summary_path = out / "dim4_spatial_clustering.csv"
    raw_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Written: {summary_path}")
    print(f"Written: {detail_path}  (per-sample detail for audit)")
    print("\nSummary  (BH-corrected; spatially_clustered = significant in ≥1 sample):")
    print(summary_df.to_string(index=False))
    print("\nInterpretation guide:")
    print("  obs_bb > exp_bb → more same-label joins than expected by chance (clustering)")
    print("  z_score > 0     → consistent direction of clustering")
    print(f"  p_bh < {args.fdr_alpha}       → spatially clustered after FDR correction")
    print("\n=== dim4_spatial_clustering.py complete ===")


if __name__ == "__main__":
    main()
