#!/usr/bin/env python3
"""
validate.py  —  Step 0 (run FIRST, before any dimension script)
═══════════════════════════════════════════════════════════════
Loads the four required input files, aligns them by cell/spot ID,
filters unassigned niches and undersized niche×sample groups,
and writes a clean AnnData + optional composition table.

Outputs
  {out_dir}/clean_data.h5ad       — obs: niche, sample [, cell_type]
                                     obsm['spatial']: (n, 2) coordinates
                                     uns['branch'], uns['run_config']
  {out_dir}/composition.csv       — (niche, sample) × cell_type proportions
                                     [Branch A or B only; needed by dim2 + dim3]
  {out_dir}/run_config.json       — all parameters + QC drop counts

Usage
  # Branch A  (single-cell, cell_type column exists in labels file)
  python validate.py --expr expr.csv --coords coords.csv \\
         --labels labels.csv --samples samples.csv \\
         --celltype-col cell_type --out-dir niche_output

  # Branch B  (spot-level, separate deconvolution proportion file)
  python validate.py --expr expr.csv --coords coords.csv \\
         --labels labels.csv --samples samples.csv \\
         --deconv deconv_proportions.csv --out-dir niche_output

  # Branch C  (no cell-type info — dims 2 and 3 will be skipped downstream)
  python validate.py --expr expr.csv --coords coords.csv \\
         --labels labels.csv --samples samples.csv --out-dir niche_output

RULE 2: files are joined by ID, NEVER by row order.
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd


# ── helpers ────────────────────────────────────────────────────────────────────

def _install(pkg):
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "-q"]
    )


def _read(path):
    p = Path(path)
    sep = "\t" if p.suffix in (".tsv", ".txt") else ","
    return pd.read_csv(p, sep=sep)


def _require_col(df, col, file_label):
    if col not in df.columns:
        sys.exit(
            f"ERROR: column '{col}' not found in {file_label} file.\n"
            f"  Available columns: {list(df.columns)}\n"
            f"  Use the appropriate --*-col flag to specify the correct name."
        )


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--expr",         required=True,  help="Expression or proportion matrix (rows=cells/spots)")
    ap.add_argument("--coords",       required=True,  help="Coordinate file")
    ap.add_argument("--labels",       required=True,  help="Niche label file")
    ap.add_argument("--samples",      required=True,  help="Sample ID file")
    ap.add_argument("--id-col",       default=None,   help="Shared ID column name (default: first column of labels file)")
    ap.add_argument("--niche-col",    default="niche",  help="Niche column in labels file (default: niche)")
    ap.add_argument("--sample-col",   default="sample", help="Sample column in samples file (default: sample)")
    ap.add_argument("--x-col",        default="x",    help="X coordinate column (default: x)")
    ap.add_argument("--y-col",        default="y",    help="Y coordinate column (default: y)")
    ap.add_argument("--celltype-col", default=None,   help="[Branch A] Cell-type column in labels file")
    ap.add_argument("--deconv",       default=None,   help="[Branch B] Spot×celltype proportion CSV")
    ap.add_argument("--min-cells",    type=int, default=20,
                    help="Min cells per niche×sample group; smaller groups dropped (default: 20)")
    ap.add_argument("--unassigned",   default="-1,NA,NaN,unassigned,unknown,none,",
                    help="Comma-separated unassigned-niche sentinels (default: -1,NA,NaN,unassigned,unknown,none,)")
    ap.add_argument("--seed",         type=int, default=42, help="Random seed recorded in run_config (default: 42)")
    ap.add_argument("--out-dir",      default="niche_output", help="Output directory (default: niche_output)")
    args = ap.parse_args()

    try:
        import anndata as ad
    except ImportError:
        print("[validate] Installing anndata..."); _install("anndata"); import anndata as ad

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    print("=" * 60)
    print("validate.py  —  loading and aligning four input files")
    print("=" * 60)

    # ── 1. Read files ──────────────────────────────────────────────────────────
    expr_df    = _read(args.expr)
    coords_df  = _read(args.coords)
    labels_df  = _read(args.labels)
    samples_df = _read(args.samples)

    id_col = args.id_col or labels_df.columns[0]
    print(f"\n[1] ID column: '{id_col}'")
    for df, name in [(expr_df,"expr"), (coords_df,"coords"), (labels_df,"labels"), (samples_df,"samples")]:
        if id_col not in df.columns:
            sys.exit(
                f"ERROR: ID column '{id_col}' not found in {name} file.\n"
                f"  Columns: {list(df.columns)}\n"
                f"  Use --id-col to specify the correct name."
            )
        df.set_index(id_col, inplace=True)

    n_raw = [len(expr_df), len(coords_df), len(labels_df), len(samples_df)]
    print(f"  Raw rows — expr:{n_raw[0]}  coords:{n_raw[1]}  labels:{n_raw[2]}  samples:{n_raw[3]}")

    # ── 2. Inner-join by ID  (RULE 2: never concat by row order) ──────────────
    common = (
        set(expr_df.index) & set(coords_df.index) & set(labels_df.index) & set(samples_df.index)
    )
    n_common = len(common)
    print(f"\n[2] Inner-join on ID: {n_common} common cells  "
          f"(dropped: {max(n_raw) - n_common})")
    if n_common == 0:
        sys.exit(
            "ERROR: no common IDs across the four files.\n"
            "  Check that all files share the same ID column (--id-col)."
        )
    common = list(common)
    expr_df    = expr_df.loc[common]
    coords_df  = coords_df.loc[common]
    labels_df  = labels_df.loc[common]
    samples_df = samples_df.loc[common]

    # ── 3. Validate required columns ──────────────────────────────────────────
    _require_col(labels_df,  args.niche_col,  "labels")
    _require_col(coords_df,  args.x_col,      "coords")
    _require_col(coords_df,  args.y_col,      "coords")
    _require_col(samples_df, args.sample_col, "samples")

    # ── 4. Clean coordinates  (PITFALL 6) ─────────────────────────────────────
    xy = coords_df[[args.x_col, args.y_col]].copy().astype(float)
    nan_mask = xy.isna().any(axis=1)
    dup_mask  = xy.duplicated()
    n_nan, n_dup = int(nan_mask.sum()), int(dup_mask.sum())
    if n_nan > 0:
        print(f"\n[3] Dropping {n_nan} cells with NaN coordinates.")
        bad_idx = xy[nan_mask].index
        for df in [expr_df, coords_df, labels_df, samples_df]:
            df.drop(bad_idx, inplace=True)
        xy = coords_df[[args.x_col, args.y_col]].astype(float)
    if n_dup > 0:
        warnings.warn(
            f"{n_dup} duplicate (x,y) pairs detected. "
            "Proceeding — graph construction may produce unexpected adjacencies."
        )

    # ── 5. Remove unassigned niche labels  (PITFALL 8) ────────────────────────
    sentinels = set(s.strip() for s in args.unassigned.split(",")) | {"-1", "", "nan"}
    niche_series = labels_df[args.niche_col].astype(str)
    keep_mask    = ~niche_series.isin(sentinels)
    n_unassigned = int((~keep_mask).sum())
    found_sentinels = sentinels & set(niche_series.unique())
    if n_unassigned > 0:
        print(f"\n[4] Dropped {n_unassigned} cells with unassigned labels "
              f"(sentinels found: {found_sentinels})")
    keep_ids = niche_series[keep_mask].index
    expr_df    = expr_df.loc[keep_ids]
    coords_df  = coords_df.loc[keep_ids]
    labels_df  = labels_df.loc[keep_ids]
    samples_df = samples_df.loc[keep_ids]
    n_after_unassigned = len(keep_ids)

    # ── 6. Build obs DataFrame + min-size filter  (PITFALL 4) ─────────────────
    obs = pd.DataFrame({
        "niche":  labels_df[args.niche_col].astype(str).values,
        "sample": samples_df[args.sample_col].astype(str).values,
    }, index=keep_ids)

    group_sizes   = obs.groupby(["niche", "sample"]).size()
    small_groups  = group_sizes[group_sizes < args.min_cells]
    n_small       = len(small_groups)
    if n_small > 0:
        print(f"\n[5] Dropping {n_small} niche×sample groups with < {args.min_cells} cells:")
        for (n, s), sz in small_groups.items():
            print(f"    niche={n}  sample={s}  cells={sz}")
        drop_pairs  = set(small_groups.index)
        keep_mask2  = ~obs.apply(lambda r: (r["niche"], r["sample"]) in drop_pairs, axis=1)
        obs         = obs[keep_mask2]
        expr_df     = expr_df.loc[obs.index]
        coords_df   = coords_df.loc[obs.index]

    n_final      = len(obs)
    n_niches     = obs["niche"].nunique()
    n_samples    = obs["sample"].nunique()
    print(f"\n[6] Final dataset: {n_final} cells | {n_niches} niches | {n_samples} samples")

    # ── 7. Determine branch + compute composition  ─────────────────────────────
    branch, cell_types, comp_df = "C", None, None

    if args.celltype_col:                                          # Branch A
        if args.celltype_col not in labels_df.columns:
            print(f"\nWARNING: --celltype-col '{args.celltype_col}' not found in labels file. "
                  "Falling back to Branch C (dims 2-3 will be skipped).")
        else:
            branch = "A"
            obs["cell_type"] = labels_df.loc[obs.index, args.celltype_col].astype(str).values
            cell_types = sorted(obs["cell_type"].unique().tolist())
            records = []
            for (niche, sample), grp in obs.groupby(["niche", "sample"]):
                ct_counts = grp["cell_type"].value_counts()
                props = {ct: ct_counts.get(ct, 0) / len(grp) for ct in cell_types}
                records.append({"niche": niche, "sample": sample, **props})
            comp_df = pd.DataFrame(records).set_index(["niche", "sample"])
            print(f"\n[7] Branch A: {len(comp_df)} niche×sample groups, "
                  f"{len(cell_types)} cell types → composition.csv")

    elif args.deconv:                                              # Branch B
        branch = "B"
        deconv_raw = _read(args.deconv)
        id_col_d   = args.id_col or deconv_raw.columns[0]
        if id_col_d in deconv_raw.columns:
            deconv_raw.set_index(id_col_d, inplace=True)
        deconv_raw = deconv_raw.loc[deconv_raw.index.intersection(obs.index)]
        cell_types = sorted(deconv_raw.columns.tolist())
        records = []
        for (niche, sample), grp in obs.groupby(["niche", "sample"]):
            grp_deconv = deconv_raw.loc[deconv_raw.index.intersection(grp.index)]
            if len(grp_deconv) == 0:
                continue
            mean_props = grp_deconv.mean().to_dict()
            records.append({"niche": niche, "sample": sample, **mean_props})
        comp_df = pd.DataFrame(records).set_index(["niche", "sample"])
        print(f"\n[7] Branch B: {len(comp_df)} niche×sample groups, "
              f"{len(cell_types)} cell types → composition.csv")

    else:                                                          # Branch C
        print("\n[7] Branch C: no cell-type info provided.")
        print("    Dims 2 and 3 will be skipped downstream.")
        print("    Re-run with --celltype-col (Branch A) or --deconv (Branch B) to enable them.")

    # ── 8. Build AnnData ───────────────────────────────────────────────────────
    spatial = coords_df.loc[obs.index, [args.x_col, args.y_col]].values.astype(float)
    adata   = ad.AnnData(
        X   = np.zeros((n_final, 1), dtype=np.float32),  # dummy — dims don't use .X
        obs = obs,
    )
    adata.obsm["spatial"]  = spatial
    adata.uns["branch"]    = branch
    adata.uns["cell_types"] = cell_types or []
    adata.uns["run_config"] = {
        "seed":                     args.seed,
        "min_cells_per_group":      args.min_cells,
        "unassigned_sentinels":     args.unassigned,
        "branch":                   branch,
        "n_raw_max":                max(n_raw),
        "n_after_join":             n_common,
        "n_after_unassigned_filter": n_after_unassigned,
        "n_final":                  n_final,
        "n_niches":                 n_niches,
        "n_samples":                n_samples,
        "n_nan_coords_dropped":     n_nan,
        "n_duplicate_coords":       n_dup,
        "n_small_groups_dropped":   n_small,
        "graph_note":               "RULE 1: spatial graphs MUST be built per-sample in dim4",
    }

    # ── 9. Write outputs ───────────────────────────────────────────────────────
    h5ad_path   = out / "clean_data.h5ad"
    config_path = out / "run_config.json"

    adata.write_h5ad(h5ad_path)
    print(f"\nWritten: {h5ad_path}")

    if comp_df is not None:
        comp_path = out / "composition.csv"
        comp_df.to_csv(comp_path)
        print(f"Written: {comp_path}")

    with open(config_path, "w") as f:
        json.dump(adata.uns["run_config"], f, indent=2)
    print(f"Written: {config_path}")

    print("\n" + "=" * 60)
    print("validate.py complete. Next steps:")
    print(f"  python dim1_prevalence.py       --data {h5ad_path}")
    print(f"  python dim2_stability.py        --data {h5ad_path}")
    print(f"  python dim3_diversity.py        --data {h5ad_path}")
    print(f"  python dim4_spatial_clustering.py --data {h5ad_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
