#!/usr/bin/env python3
"""
validate.py — Prerequisite Step: Validate & Load Raw Spatial Omics Data
========================================================================
Reads the five raw input files, inner-joins them by cell/spot ID (Rule 2),
cleans coordinates, drops unassigned niche labels, applies minimum niche-size
filter, and writes:

  Outputs
  -------
  clean_data.h5ad    AnnData consumed by every downstream dim*.py script
  composition.csv    Cell-type composition per cell/spot (Branch A / B only)
  run_config.json    Reproducibility record (params, versions, drop counts)

Platform branches
-----------------
  Branch A  High-resolution (CosMx / Xenium / MERFISH): cell-type annotation
            column supplies the composition axis.
  Branch B  Low-resolution WITH proportion matrix (Visium / SlideSeq):
            spot×cell proportion matrix supplies the composition axis.
  Branch C  Low-resolution WITHOUT proportion matrix: dims 2 & 3 unavailable;
            composition.csv is NOT written; script exits without error.

Usage
-----
  python validate.py \\
      --platform  high \\
      --matrix    data/counts.h5ad \\
      --celltype  data/cell_types.csv \\
      --coords    data/coords.csv \\
      --niches    data/niche_labels.csv \\
      --samples   data/sample_ids.csv \\
      --outdir    results/

  python validate.py \\
      --platform   low \\
      --matrix     data/spots.csv \\
      --proportion data/deconv_props.csv \\
      --coords     data/coords.csv \\
      --niches     data/niche_labels.csv \\
      --samples    data/sample_ids.csv \\
      --outdir     results/

Run BEFORE any dim*.py script.
"""

import argparse
import importlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

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
        description="Validate & load raw spatial omics files for the niche-description pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--platform", choices=["high", "low"], required=True,
        help="'high' = single-cell (CosMx/Xenium/MERFISH); 'low' = spot-based (Visium/SlideSeq)",
    )
    # ── Input files ────────────────────────────────────────────────────────────
    p.add_argument("--matrix", required=True,
                   help="Cell×gene (high-res) or spot×gene (low-res) count matrix "
                        "(.csv / .tsv / .h5ad)")
    p.add_argument("--celltype", default=None,
                   help="[high-res] Cell-type annotation file (first column = label, "
                        "index = cell ID)")
    p.add_argument("--proportion", default=None,
                   help="[low-res] Spot×cell proportion matrix (.csv / .tsv); "
                        "rows sum to ≈ 1; index = spot ID; columns = cell-type names")
    p.add_argument("--coords", required=True,
                   help="Coordinate file; index = cell/spot ID; columns must include 'x' and 'y' "
                        "(or first two columns are used)")
    p.add_argument("--niches", required=True,
                   help="Niche-label file; index = cell/spot ID; first column = niche label")
    p.add_argument("--samples", required=True,
                   help="Sample-ID file; index = cell/spot ID; first column = sample ID")
    # ── Output & run params ────────────────────────────────────────────────────
    p.add_argument("--outdir", default=".",
                   help="Output directory (default: current directory)")
    p.add_argument("--min-niche-size", type=int, default=20,
                   help="Minimum cells/spots per niche×sample group to keep (default: 20)")
    p.add_argument(
        "--unassigned-labels", nargs="+",
        default=["-1", "NA", "na", "NaN", "nan", "unassigned", "Unassigned", ""],
        help="Niche label values to treat as unassigned and drop",
    )
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed — fixed, recorded in run_config.json (default: 42)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────
def read_tabular(path: str) -> pd.DataFrame:
    """
    Auto-detect CSV / TSV / whitespace-separated file.
    First column is always used as the index (cell/spot ID).
    Returns a DataFrame with string index.
    """
    p = Path(path)
    sep = "\t" if p.suffix in (".tsv", ".txt") else ","
    try:
        df = pd.read_csv(p, sep=sep, index_col=0)
    except Exception:
        # Fallback: let pandas sniff the separator
        df = pd.read_csv(p, sep=None, engine="python", index_col=0)
    df.index = df.index.astype(str)
    return df


def load_matrix(path: str) -> ad.AnnData:
    """
    Load expression matrix.
    Accepts .h5ad directly, or any tabular file (rows=cells, cols=genes).
    Always returns an AnnData with string obs index.
    """
    p = Path(path)
    if p.suffix == ".h5ad":
        adata = ad.read_h5ad(p)
        adata.obs.index = adata.obs.index.astype(str)
        return adata
    df = read_tabular(path)
    log.info("  Tabular matrix: %d cells × %d genes", *df.shape)
    adata = ad.AnnData(
        X=df.values.astype(np.float32),
        obs=pd.DataFrame(index=df.index),
        var=pd.DataFrame(index=df.columns),
    )
    return adata


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)          # legacy API
    rng = np.random.default_rng(args.seed)  # noqa: F841 — seed is captured in config

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    drop_counts: dict = {}

    # ── 1. Load expression matrix ──────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Step 1 — Loading expression matrix: %s", args.matrix)
    adata = load_matrix(args.matrix)
    n_initial = adata.n_obs
    log.info("  Loaded: %d cells/spots × %d features", n_initial, adata.n_vars)

    # ── 2. Load auxiliary tabular files ────────────────────────────────────────
    log.info("Step 2 — Loading auxiliary files")

    log.info("  Coordinates: %s", args.coords)
    coords_df = read_tabular(args.coords)
    if "x" in coords_df.columns and "y" in coords_df.columns:
        coords_df = coords_df[["x", "y"]]
    else:
        coords_df = coords_df.iloc[:, :2].copy()
        coords_df.columns = ["x", "y"]
        log.warning("  'x'/'y' column names not found — using first two columns as x, y.")

    log.info("  Niche labels: %s", args.niches)
    niches_df = read_tabular(args.niches)
    niches_df = niches_df.iloc[:, [0]].rename(columns={niches_df.columns[0]: "niche"})
    niches_df["niche"] = niches_df["niche"].astype(str)

    log.info("  Sample IDs: %s", args.samples)
    samples_df = read_tabular(args.samples)
    samples_df = samples_df.iloc[:, [0]].rename(columns={samples_df.columns[0]: "sample"})
    samples_df["sample"] = samples_df["sample"].astype(str)

    # ── Platform-specific composition file ────────────────────────────────────
    ct_df = None
    proportion_df = None

    if args.platform == "high":
        if args.celltype:
            log.info("  Cell-type annotation [high-res]: %s", args.celltype)
            ct_df = read_tabular(args.celltype)
            ct_df = ct_df.iloc[:, [0]].rename(columns={ct_df.columns[0]: "cell_type"})
            ct_df["cell_type"] = ct_df["cell_type"].astype(str)
        else:
            log.warning(
                "  --celltype not provided for high-res platform. "
                "Branch C: dims 2 and 3 will be UNAVAILABLE."
            )
    else:  # low
        if args.proportion:
            log.info("  Proportion matrix [low-res]: %s", args.proportion)
            proportion_df = read_tabular(args.proportion)
            proportion_df = proportion_df.astype(float)
            row_sums = proportion_df.sum(axis=1)
            n_bad = int((np.abs(row_sums - 1.0) > 0.01).sum())
            if n_bad > 0:
                log.warning(
                    "  %d / %d spots deviate from row-sum = 1.0 by > 0.01. "
                    "Verify your deconvolution output.",
                    n_bad, len(proportion_df),
                )
        else:
            log.warning(
                "  --proportion not provided for low-res platform. "
                "Branch C: dims 2 and 3 will be UNAVAILABLE."
            )

    # ── 3. Inner-join all files by ID (Rule 2 — NEVER assume row order) ────────
    log.info("Step 3 — Inner-joining all files by cell/spot ID")
    common = adata.obs.index

    n_before = len(common)
    common = common.intersection(coords_df.index)
    drop_counts["coords_id_mismatch"] = n_before - len(common)

    n_before = len(common)
    common = common.intersection(niches_df.index)
    drop_counts["niches_id_mismatch"] = n_before - len(common)

    n_before = len(common)
    common = common.intersection(samples_df.index)
    drop_counts["samples_id_mismatch"] = n_before - len(common)

    if ct_df is not None:
        n_before = len(common)
        common = common.intersection(ct_df.index)
        drop_counts["celltype_id_mismatch"] = n_before - len(common)

    if proportion_df is not None:
        n_before = len(common)
        common = common.intersection(proportion_df.index)
        drop_counts["proportion_id_mismatch"] = n_before - len(common)

    total_dropped_join = n_initial - len(common)
    if total_dropped_join > 0:
        log.warning(
            "  ID-mismatch total: %d / %d cells/spots dropped across all joins.",
            total_dropped_join, n_initial,
        )
        for key, val in drop_counts.items():
            if val > 0:
                log.warning("    %s: %d", key, val)
    else:
        log.info("  All IDs matched across files — 0 rows dropped.")

    common_list = list(common)
    adata = adata[common_list].copy()

    # ── 4. Attach obs columns ──────────────────────────────────────────────────
    adata.obs["niche"] = niches_df.loc[common_list, "niche"].values
    adata.obs["sample"] = samples_df.loc[common_list, "sample"].values

    if ct_df is not None:
        adata.obs["cell_type"] = ct_df.loc[common_list, "cell_type"].values

    if proportion_df is not None:
        prop_aligned = proportion_df.loc[common_list].values.astype(np.float32)
        adata.obsm["cell_proportions"] = prop_aligned
        adata.uns["cell_type_names"] = list(proportion_df.columns)

    adata.obsm["spatial"] = coords_df.loc[common_list, ["x", "y"]].values.astype(np.float64)

    # ── 5. Drop unassigned niche labels ───────────────────────────────────────
    log.info("Step 4 — Filtering: dropping unassigned niche labels")
    unassigned_set = set(str(v) for v in args.unassigned_labels)
    assigned_mask = ~adata.obs["niche"].isin(unassigned_set)
    n_unassigned = int((~assigned_mask).sum())
    drop_counts["unassigned_niche_labels"] = n_unassigned
    if n_unassigned > 0:
        log.warning(
            "  Dropped %d cells/spots with unassigned niche label "
            "(values: %s).",
            n_unassigned,
            ", ".join(sorted(adata.obs["niche"][~assigned_mask].unique())),
        )
        adata = adata[assigned_mask].copy()
    else:
        log.info("  No unassigned niche labels found.")

    # ── 6. Clean spatial coordinates ──────────────────────────────────────────
    log.info("Step 5 — Cleaning spatial coordinates")
    xy = adata.obsm["spatial"]

    # NaN coordinates
    nan_rows = np.any(np.isnan(xy), axis=1)
    n_nan = int(nan_rows.sum())
    drop_counts["nan_coordinates"] = n_nan
    if n_nan > 0:
        log.warning("  Dropped %d cells/spots with NaN coordinates.", n_nan)
        adata = adata[~nan_rows].copy()
        xy = adata.obsm["spatial"]

    # Duplicate coordinates (warn only — valid in regular spot arrays)
    _, unique_idx = np.unique(xy, axis=0, return_index=True)
    n_dup = len(xy) - len(unique_idx)
    drop_counts["duplicate_coordinate_pairs_warned"] = n_dup
    if n_dup > 0:
        log.warning(
            "  %d duplicate (x, y) coordinate pairs detected. "
            "Inspect if unexpected (OK for regular spot grids).",
            n_dup,
        )

    # IQR-based outlier warning (not dropped — let the user decide)
    for axis_i, axis_name in enumerate(["x", "y"]):
        col = xy[:, axis_i]
        q1, q3 = np.percentile(col, [25, 75])
        iqr = q3 - q1
        n_out = int(((col < q1 - 5 * iqr) | (col > q3 + 5 * iqr)).sum())
        if n_out > 0:
            log.warning(
                "  %d potential %s-coordinate outliers (>5×IQR). "
                "Verify before running dim 4 — may distort the spatial graph.",
                n_out, axis_name,
            )

    # ── 7. Minimum niche×sample size filter ───────────────────────────────────
    log.info("Step 6 — Minimum niche×sample size filter (min = %d)", args.min_niche_size)
    group_sizes = adata.obs.groupby(["niche", "sample"], observed=True).size()
    small_groups = group_sizes[group_sizes < args.min_niche_size]

    n_filtered = 0
    if not small_groups.empty:
        keep_mask = pd.Series(True, index=adata.obs.index)
        for (niche_label, sample_label), cnt in small_groups.items():
            bad = (adata.obs["niche"] == niche_label) & (adata.obs["sample"] == sample_label)
            keep_mask[bad] = False
            log.warning(
                "  Removed niche=%-12s  sample=%-12s  n=%d  (< %d)",
                niche_label, sample_label, cnt, args.min_niche_size,
            )
        n_filtered = int((~keep_mask).sum())
        drop_counts["min_size_filter"] = n_filtered
        adata = adata[keep_mask.values].copy()
        log.warning(
            "  Total cells/spots removed by size filter: %d. "
            "Retained: %d.",
            n_filtered, adata.n_obs,
        )
    else:
        log.info(
            "  All niche×sample groups meet the minimum size (%d). "
            "No cells filtered.",
            args.min_niche_size,
        )

    # ── 8. Determine branch & build composition.csv ───────────────────────────
    log.info("Step 7 — Determining platform branch and building composition.csv")
    branch = "C"

    if args.platform == "high" and "cell_type" in adata.obs.columns:
        branch = "A"
        log.info("  Branch A (high-res, cell-type annotation present).")
        comp_df = adata.obs[["cell_type"]].copy()
        comp_df.to_csv(outdir / "composition.csv")
        log.info("  composition.csv written (%d rows, 1 column: cell_type).", len(comp_df))

    elif args.platform == "low" and "cell_proportions" in adata.obsm:
        branch = "B"
        cell_type_names = adata.uns.get(
            "cell_type_names",
            [f"cell_type_{i}" for i in range(adata.obsm["cell_proportions"].shape[1])],
        )
        comp_df = pd.DataFrame(
            adata.obsm["cell_proportions"],
            index=adata.obs.index,
            columns=cell_type_names,
        )
        comp_df.to_csv(outdir / "composition.csv")
        log.info(
            "  Branch B (low-res, proportion matrix present). "
            "composition.csv written (%d spots × %d cell types).",
            *comp_df.shape,
        )

    else:
        log.warning(
            "  Branch C — no composition axis available. "
            "composition.csv NOT written. "
            "Dims 2 and 3 will be skipped downstream."
        )

    # ── 9. Store metadata & write clean_data.h5ad ──────────────────────────────
    adata.uns["branch"] = branch
    adata.uns["platform"] = args.platform
    adata.uns["seed"] = args.seed
    adata.uns["min_niche_size"] = args.min_niche_size

    h5ad_path = outdir / "clean_data.h5ad"
    log.info("Step 8 — Writing %s (%d cells/spots)", h5ad_path, adata.n_obs)
    adata.write_h5ad(h5ad_path)

    # ── 10. Write run_config.json ──────────────────────────────────────────────
    pkg_versions: dict = {}
    for pkg in ["anndata", "numpy", "pandas", "scipy", "statsmodels"]:
        try:
            pkg_versions[pkg] = importlib.import_module(pkg).__version__
        except Exception:
            pkg_versions[pkg] = "unavailable"

    config = {
        "run_timestamp": datetime.now().isoformat(),
        "platform": args.platform,
        "branch": branch,
        "seed": args.seed,
        "min_niche_size": args.min_niche_size,
        "unassigned_labels": args.unassigned_labels,
        "n_initial": n_initial,
        "n_final": adata.n_obs,
        "n_niches": int(adata.obs["niche"].nunique()),
        "n_samples": int(adata.obs["sample"].nunique()),
        "niche_labels": sorted(adata.obs["niche"].unique().tolist()),
        "sample_labels": sorted(adata.obs["sample"].unique().tolist()),
        "drop_counts": drop_counts,
        "input_files": {
            "matrix": str(args.matrix),
            "celltype": str(args.celltype),
            "proportion": str(args.proportion),
            "coords": str(args.coords),
            "niches": str(args.niches),
            "samples": str(args.samples),
        },
        "package_versions": pkg_versions,
    }
    config_path = outdir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    log.info("run_config.json written.")

    # ── Summary ────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Validation complete.")
    log.info("  Branch  : %s | Platform: %s", branch, args.platform)
    log.info("  Retained: %d / %d cells/spots", adata.n_obs, n_initial)
    log.info("  Niches  : %d", adata.obs["niche"].nunique())
    log.info("  Samples : %d", adata.obs["sample"].nunique())
    log.info("  Output  : %s", outdir.resolve())
    if branch == "C":
        log.info("  ⚠  Dims 2 and 3 require a composition axis — run only dims 1 and 4.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
