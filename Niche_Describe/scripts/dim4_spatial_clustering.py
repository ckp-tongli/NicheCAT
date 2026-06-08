#!/usr/bin/env python3
"""
dim4_spatial_clustering.py — Dimension 4: Single-Niche Spatial Clustering
==========================================================================
Is each niche spatially clumped within tissue, or scattered?

Reads
-----
  clean_data.h5ad     (produced by validate.py)

Writes
------
  dim4_spatial_clustering.csv          One row per niche (aggregated summary)
  dim4_spatial_clustering_detail.csv   One row per niche × sample (raw results)

Method (per niche, per sample — RULE 1: NEVER pool samples)
-----------------------------------------------------------
  1. Binarize: cells in the target niche → 1; all other cells in the sample → 0.
     Niche labels are CATEGORICAL; this script never feeds them into Moran's I /
     Geary's C / Getis-Ord G (Rule 3 — those tests require a continuous variable).

  2. Build a symmetric spatial graph using one of three methods:
       knn      k nearest neighbours (default: k=6)
       radius   fixed radius in coordinate units
       delaunay Delaunay triangulation (good for irregular single-cell layouts)
     Implemented with scipy.spatial.cKDTree / scipy.spatial.Delaunay —
     no squidpy, esda, or libpysal required.

  3. Count observed BB (1–1) join count = Σ_{(i,j)∈E} yᵢ · yⱼ

  4. Permutation test (n_permutations=999 by default; seed from validate.py):
     Randomly shuffle the binary y vector n times (preserving total count),
     compute BB joins each time, derive p-value as:
         p = (# permutations with BB_perm ≥ BB_obs + 1) / (n_permutations + 1)
     One-sided test (we test for clustering, not dispersion).

  5. BH/FDR correction across the FULL niche × sample test grid.

  6. Aggregate per-sample results to a per-niche summary:
       fraction_samples_clustered = # significant samples / # tested samples
       spatial_clustering_label   = 'clustered' if ≥ 50% samples are significant

Usage
-----
  python dim4_spatial_clustering.py --indir results/ --outdir results/
  python dim4_spatial_clustering.py --indir results/ --graph-type radius --radius 50
  python dim4_spatial_clustering.py --indir results/ --graph-type delaunay

Run after validate.py.
"""

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree
from statsmodels.stats.multitest import multipletests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dimension 4 — Single-niche spatial clustering (Join Count + permutation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--indir", default=".",
                   help="Directory containing clean_data.h5ad (default: .)")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: same as --indir)")
    p.add_argument(
        "--graph-type", choices=["knn", "radius", "delaunay"], default="knn",
        help="Spatial graph construction method (default: knn)",
    )
    p.add_argument(
        "--k", type=int, default=6,
        help="Number of nearest neighbours for kNN graph (default: 6, "
             "the standard 6-neighbour hexagonal grid default)",
    )
    p.add_argument(
        "--radius", type=float, default=None,
        help="Fixed search radius in coordinate units (required when "
             "--graph-type=radius)",
    )
    p.add_argument(
        "--n-permutations", type=int, default=999,
        help="Number of permutations for the Join Count p-value (default: 999)",
    )
    p.add_argument(
        "--fdr-alpha", type=float, default=0.05,
        help="FDR significance threshold for BH correction (default: 0.05)",
    )
    p.add_argument(
        "--clustered-fraction", type=float, default=0.50,
        help="Fraction of samples that must be significant for a niche to be "
             "labelled 'clustered' in the summary (default: 0.50)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed override (default: use seed stored in clean_data.h5ad)",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Spatial graph builders
# ──────────────────────────────────────────────────────────────────────────────
def build_knn_graph(coords: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Symmetric kNN graph. Each node connects to its k nearest neighbours."""
    n = len(coords)
    k_actual = min(k + 1, n)   # +1 because query returns self as rank-0 hit
    tree = cKDTree(coords)
    _, idxs = tree.query(coords, k=k_actual)
    edges: set[tuple[int, int]] = set()
    for i, nbrs in enumerate(idxs):
        for j in nbrs:
            if j != i:
                edges.add((min(int(i), int(j)), max(int(i), int(j))))
    return list(edges)


def build_radius_graph(coords: np.ndarray, radius: float) -> list[tuple[int, int]]:
    """Symmetric radius graph. Edges connect all pairs within `radius`."""
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=radius)
    return [(min(int(a), int(b)), max(int(a), int(b))) for a, b in pairs]


def build_delaunay_graph(coords: np.ndarray) -> list[tuple[int, int]]:
    """
    Delaunay triangulation graph.
    Falls back to kNN-6 for degenerate configurations (< 3 points, collinear).
    """
    if len(coords) < 3:
        return []
    try:
        tri = Delaunay(coords)
        edges: set[tuple[int, int]] = set()
        for simplex in tri.simplices:
            for a, b in combinations(simplex, 2):
                edges.add((min(int(a), int(b)), max(int(a), int(b))))
        return list(edges)
    except Exception as exc:
        log.warning(
            "    Delaunay triangulation failed (%s). Falling back to kNN-6.", exc
        )
        return build_knn_graph(coords, k=6)


def build_graph(
    coords: np.ndarray,
    graph_type: str,
    k: int = 6,
    radius: Optional[float] = None,
) -> list[tuple[int, int]]:
    """Dispatch to the correct graph builder."""
    if graph_type == "knn":
        return build_knn_graph(coords, k)
    elif graph_type == "radius":
        if radius is None:
            raise ValueError("radius must be set when graph_type='radius'")
        return build_radius_graph(coords, radius)
    elif graph_type == "delaunay":
        return build_delaunay_graph(coords)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Join Count statistic
# ──────────────────────────────────────────────────────────────────────────────
def count_bb_joins(y: np.ndarray, edge_a: np.ndarray, edge_b: np.ndarray) -> int:
    """
    Count BB (1–1) join count statistic.

    y      : binary array of length n_cells (1 = target niche, 0 = other).
    edge_a : array of first endpoints (local integer index).
    edge_b : array of second endpoints (local integer index).
    Both endpoint arrays are pre-computed from the edge list for speed.
    """
    return int(np.sum(y[edge_a] * y[edge_b]))


def run_permutation_test(
    y: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    n_perms: int,
    rng: np.random.Generator,
) -> tuple[int, float, float]:
    """
    Permutation test for the BB join count.

    Returns
    -------
    jc_obs   : observed BB join count
    z_score  : (jc_obs − mean_perm) / std_perm   (NaN if std_perm = 0)
    p_value  : (# perms with BB_perm ≥ jc_obs + 1) / (n_perms + 1)
               One-sided, testing for clustering (excess same-label joins).
    """
    jc_obs = count_bb_joins(y, edge_a, edge_b)

    if len(edge_a) == 0:
        return jc_obs, np.nan, np.nan

    # Permute labels (preserves marginal sum of y)
    y_perm = y.copy()
    perm_jc = np.empty(n_perms, dtype=float)
    for i in range(n_perms):
        rng.shuffle(y_perm)
        perm_jc[i] = count_bb_joins(y_perm, edge_a, edge_b)

    mu = perm_jc.mean()
    sigma = perm_jc.std()
    z = float((jc_obs - mu) / sigma) if sigma > 0 else np.nan

    # One-sided p-value: proportion of permutations ≥ observed
    p = float((perm_jc >= jc_obs).sum() + 1) / (n_perms + 1)

    return jc_obs, z, p


# ──────────────────────────────────────────────────────────────────────────────
# Optional type hint helper (Python 3.9 compatibility)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from typing import Optional
except ImportError:
    Optional = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    indir = Path(args.indir)
    outdir = Path(args.outdir) if args.outdir else indir
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Validate CLI ──────────────────────────────────────────────────────────
    if args.graph_type == "radius" and args.radius is None:
        log.error("--radius must be provided when --graph-type=radius.")
        sys.exit(1)

    # ── Load AnnData ──────────────────────────────────────────────────────────
    h5ad_path = indir / "clean_data.h5ad"
    if not h5ad_path.exists():
        log.error(
            "clean_data.h5ad not found in %s. Run validate.py first.", indir
        )
        sys.exit(1)

    log.info("Loading %s", h5ad_path)
    adata = ad.read_h5ad(h5ad_path)

    # Resolve random seed
    stored_seed = adata.uns.get("seed", 42)
    seed = args.seed if args.seed is not None else stored_seed
    rng = np.random.default_rng(seed)
    log.info("  Random seed: %d  (source: %s)",
             seed, "CLI --seed" if args.seed is not None else "clean_data.h5ad")

    obs = adata.obs[["niche", "sample"]].copy()
    coords_all: np.ndarray = adata.obsm["spatial"]   # (n_cells, 2), aligned with obs

    niches = sorted(obs["niche"].unique())
    samples = sorted(obs["sample"].unique())
    log.info("  %d niches | %d samples", len(niches), len(samples))
    log.info(
        "  Graph: %s%s  |  Permutations: %d  |  FDR α: %.2f",
        args.graph_type,
        f"  k={args.k}" if args.graph_type == "knn"
        else f"  r={args.radius}" if args.graph_type == "radius"
        else "",
        args.n_permutations,
        args.fdr_alpha,
    )

    # ── Pre-build one graph per sample (Rule 1 — NEVER pool samples) ──────────
    # Building the graph depends only on the sample's coordinates, not on the
    # niche being tested — so we reuse the same graph for all niches in a sample.
    log.info("Pre-building spatial graphs (one per sample)…")
    sample_graph_cache: dict = {}

    for sample in samples:
        sample_bool = (obs["sample"] == sample).values        # boolean mask over obs
        n_sample = int(sample_bool.sum())

        if n_sample < 3:
            log.warning(
                "  Sample %-15s has only %d cells/spots — skipping (need ≥ 3).",
                sample, n_sample,
            )
            continue

        s_coords = coords_all[sample_bool]                    # local coords (0-indexed)
        edges = build_graph(s_coords, args.graph_type, k=args.k, radius=args.radius)

        if not edges:
            log.warning(
                "  Sample %-15s: graph has 0 edges — skipping.", sample
            )
            continue

        ea = np.array([e[0] for e in edges], dtype=np.intp)
        eb = np.array([e[1] for e in edges], dtype=np.intp)

        # Store binary niche membership for efficient lookup
        sample_niche_arr = obs["niche"].values[sample_bool]   # niche label per cell

        sample_graph_cache[sample] = {
            "bool_mask": sample_bool,
            "niche_arr": sample_niche_arr,
            "n": n_sample,
            "n_edges": len(edges),
            "edge_a": ea,
            "edge_b": eb,
        }

    log.info(
        "  Graphs built for %d / %d samples.", len(sample_graph_cache), len(samples)
    )

    # ── Run Join Count test: per niche × per sample ────────────────────────────
    log.info("Running Join Count permutation tests…")
    detail_rows: list[dict] = []

    for niche_i, niche in enumerate(niches):
        log.info(
            "  [%d/%d] Niche: %s", niche_i + 1, len(niches), niche
        )

        for sample, sg in sample_graph_cache.items():
            y = (sg["niche_arr"] == niche).astype(np.int8)
            n1 = int(y.sum())   # cells in this niche within this sample

            base = {
                "niche": niche,
                "sample": sample,
                "n_cells_in_sample": sg["n"],
                "n_niche_cells": n1,
                "n_edges": sg["n_edges"],
            }

            if n1 == 0:
                # Niche absent from this sample — skip (not an error)
                continue

            if n1 == sg["n"]:
                # All cells in sample belong to this niche; degenerate — trivially clustered
                detail_rows.append(
                    {
                        **base,
                        "JC_observed": sg["n_edges"],   # all joins are BB
                        "JC_z": np.nan,
                        "p_value_raw": np.nan,
                        "note": "all cells in sample belong to this niche — degenerate",
                    }
                )
                continue

            jc_obs, z, p = run_permutation_test(
                y, sg["edge_a"], sg["edge_b"], args.n_permutations, rng
            )

            detail_rows.append(
                {
                    **base,
                    "JC_observed": jc_obs,
                    "JC_z": round(z, 3) if not np.isnan(z) else np.nan,
                    "p_value_raw": round(p, 4) if not np.isnan(p) else np.nan,
                    "note": "",
                }
            )

    detail_df = pd.DataFrame(detail_rows)

    if detail_df.empty:
        log.error(
            "No valid niche×sample tests were computed. "
            "Check that samples have ≥ 3 cells and graphs have edges."
        )
        sys.exit(1)

    # ── BH/FDR correction across the FULL niche × sample test grid ────────────
    # Pitfall 10: correct ALL p-values jointly, not per-niche separately.
    valid_pval_mask = detail_df["p_value_raw"].notna()
    n_tests = int(valid_pval_mask.sum())
    log.info("Applying BH/FDR correction over %d tests…", n_tests)

    detail_df["p_value_fdr"] = np.nan
    if n_tests > 0:
        pvals = detail_df.loc[valid_pval_mask, "p_value_raw"].values
        _, pvals_fdr, _, _ = multipletests(pvals, alpha=args.fdr_alpha, method="fdr_bh")
        detail_df.loc[valid_pval_mask, "p_value_fdr"] = np.round(pvals_fdr, 4)

    detail_df["significant_fdr"] = detail_df["p_value_fdr"] < args.fdr_alpha

    # Write detail CSV
    detail_out = outdir / "dim4_spatial_clustering_detail.csv"
    detail_df.to_csv(detail_out, index=False)
    log.info("Written: %s  (%d rows)", detail_out, len(detail_df))

    # ── Aggregate to per-niche summary ─────────────────────────────────────────
    summary_rows: list[dict] = []

    for niche, grp in detail_df.groupby("niche", sort=True):
        tested = grp[grp["p_value_fdr"].notna()]
        n_tested = len(tested)
        n_sig = int(tested["significant_fdr"].sum())
        frac_clustered = round(n_sig / n_tested, 4) if n_tested > 0 else np.nan
        median_z = round(float(tested["JC_z"].median()), 3) \
            if not tested["JC_z"].isna().all() else np.nan
        mean_jc = round(float(tested["JC_observed"].mean()), 2) \
            if not tested["JC_observed"].isna().all() else np.nan

        spatial_label = (
            "clustered"
            if n_tested > 0 and frac_clustered >= args.clustered_fraction
            else "scattered"
            if n_tested > 0
            else "undetermined"
        )

        reading = (
            f"{n_sig} / {n_tested} samples show significant BB clustering "
            f"(FDR < {args.fdr_alpha}); "
            f"median Z = {median_z}; "
            f"label = {spatial_label}."
        ) if n_tested > 0 else "No testable samples."

        summary_rows.append(
            {
                "niche": niche,
                "n_samples_tested": n_tested,
                "n_samples_significant": n_sig,
                "fraction_samples_clustered": frac_clustered,
                "median_JC_z": median_z,
                "mean_JC_observed": mean_jc,
                "spatial_clustering_label": spatial_label,
                "reading": reading,
            }
        )

    summary_df = (
        pd.DataFrame(summary_rows)
        .set_index("niche")
        .sort_values("fraction_samples_clustered", ascending=False, na_position="last")
    )

    summary_out = outdir / "dim4_spatial_clustering.csv"
    summary_df.to_csv(summary_out)
    log.info("Written: %s  (%d niches)", summary_out, len(summary_df))

    # ── Console summary ────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Dimension 4 — Single-Niche Spatial Clustering (Join Count)")
    log.info(
        "  Graph: %-10s | k/r: %s | Permutations: %d | Seed: %d",
        args.graph_type,
        args.k if args.graph_type == "knn" else args.radius,
        args.n_permutations,
        seed,
    )
    log.info(
        "  FDR α = %.2f  |  Clustered-fraction threshold = %.2f",
        args.fdr_alpha, args.clustered_fraction,
    )
    log.info(
        "  Clustered niches  (≥ %.0f%% samples significant): %d / %d",
        args.clustered_fraction * 100,
        (summary_df["spatial_clustering_label"] == "clustered").sum(),
        len(summary_df),
    )
    log.info(
        "  Scattered niches: %d",
        (summary_df["spatial_clustering_label"] == "scattered").sum(),
    )
    log.info(
        "  Undetermined (no testable samples): %d",
        (summary_df["spatial_clustering_label"] == "undetermined").sum(),
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
