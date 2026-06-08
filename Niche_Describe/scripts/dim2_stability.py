#!/usr/bin/env python3
"""
dim2_stability.py — Dimension 2: Cross-sample Stability
=========================================================
Is the niche's cell-type composition consistent from sample to sample?

Reads
-----
  clean_data.h5ad     (produced by validate.py)
  composition.csv     (produced by validate.py, Branch A or B only)

Writes
------
  dim2_stability.csv

Method
------
  For each niche that passes prevalence/size filters and appears in ≥ 2 samples:

  1. Build a per-sample composition vector aligned to the GLOBAL cell-type
     vocabulary (missing types filled as 0 — mandatory vocabulary alignment).

     Branch A (high-res): frequency distribution of cell-type labels among
                          cells assigned to the niche in that sample.
     Branch B (low-res):  size-weighted mean of the spot×cell proportion
                          vectors for spots assigned to the niche in that sample.

  2. Compute the niche's global centroid as the equal-weight mean of all
     per-sample vectors (robust O(n) approach vs all-pairs O(n²)).

  3. Compute JSD(per-sample vector ‖ centroid) for each sample.
     JSD is the square root of the Jensen–Shannon divergence to give a
     distance metric in [0, 1].

  4. Report: mean_JSD, max_JSD, min_JSD, std_JSD across samples.
     mean_JSD < 0.3 → high stability (default threshold, tunable).

  Single-sample niches: stability is UNDEFINED → reported as N/A, NOT 0.

Exits clearly (code 2) if composition.csv is absent (Branch C).

Usage
-----
  python dim2_stability.py --indir results/ --outdir results/

Run after validate.py.
"""

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.special import rel_entr   # element-wise KL divergence (stable)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_EPS = 1e-10   # smoothing to prevent log(0) in JSD


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dimension 2 — Cross-sample stability (Jensen–Shannon Divergence).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--indir", default=".",
                   help="Directory containing clean_data.h5ad and composition.csv "
                        "(default: .)")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: same as --indir)")
    p.add_argument("--jsd-threshold", type=float, default=0.30,
                   help="mean-JSD threshold: below → 'high' stability (default: 0.30)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# JSD
# ──────────────────────────────────────────────────────────────────────────────
def js_divergence(p_vec: np.ndarray, q_vec: np.ndarray) -> float:
    """
    Jensen–Shannon divergence between two non-negative vectors.
    Both vectors are normalised internally (so they don't need to sum to 1
    before calling this function).
    Returns a value in [0, ln 2] (natural-log base); divide by ln 2 for bits.
    Returns 0.0 when both vectors are identical; NaN on degenerate input.
    """
    p = np.asarray(p_vec, dtype=float) + _EPS
    q = np.asarray(q_vec, dtype=float) + _EPS
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * rel_entr(p, m).sum() + 0.5 * rel_entr(q, m).sum())


def jsd_distance(p_vec: np.ndarray, q_vec: np.ndarray) -> float:
    """
    Jensen–Shannon distance = sqrt(JSD).
    Metric in [0, 1] when JSD is normalised to [0, ln 2].
    We keep the raw JSD (not distance) in the output to match the skill spec.
    """
    return float(np.sqrt(max(0.0, js_divergence(p_vec, q_vec))))


# ──────────────────────────────────────────────────────────────────────────────
# Composition vector builders (both branches)
# ──────────────────────────────────────────────────────────────────────────────
def freq_vector_from_labels(labels: np.ndarray, vocab: list) -> np.ndarray:
    """
    Branch A: build a frequency vector over `vocab` from a categorical label
    array.  Missing types receive weight 0.  The result sums to 1.
    """
    counts = pd.Series(labels).value_counts()
    vec = np.array([counts.get(ct, 0) for ct in vocab], dtype=float)
    total = vec.sum()
    return vec / total if total > 0 else vec


def mean_proportion_vector(prop_matrix: np.ndarray) -> np.ndarray:
    """
    Branch B: size-weighted mean of a (n_spots × n_types) proportion matrix.
    Each row is already a proportion vector summing to ≈ 1.
    The mean is NOT re-normalized — it already sums to ≈ 1.
    """
    return prop_matrix.mean(axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    indir = Path(args.indir)
    outdir = Path(args.outdir) if args.outdir else indir
    outdir.mkdir(parents=True, exist_ok=True)

    h5ad_path = indir / "clean_data.h5ad"
    comp_path = indir / "composition.csv"

    # ── Guard: require clean_data.h5ad ────────────────────────────────────────
    if not h5ad_path.exists():
        log.error(
            "clean_data.h5ad not found in %s. Run validate.py first.", indir
        )
        sys.exit(1)

    # ── Guard: require composition.csv (Branch A or B) ────────────────────────
    if not comp_path.exists():
        log.error(
            "composition.csv not found in %s.\n"
            "This is Branch C (no cell-type annotation or proportion matrix).\n"
            "Dimension 2 requires a composition axis to measure cross-sample "
            "consistency.\n"
            "Action: supply --celltype (high-res) or --proportion (low-res) "
            "to validate.py and re-run.",
            indir,
        )
        sys.exit(2)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading %s", h5ad_path)
    adata = ad.read_h5ad(h5ad_path)
    branch = adata.uns.get("branch", "A")
    is_proportion = branch == "B"
    obs = adata.obs[["niche", "sample"]].copy()

    log.info("Loading %s  (Branch %s)", comp_path, branch)
    comp_df = pd.read_csv(comp_path, index_col=0)
    comp_df.index = comp_df.index.astype(str)

    # Restrict to cells present in both adata and comp_df (should be identical
    # after validate.py, but guard against manual edits)
    shared_ids = obs.index.intersection(comp_df.index)
    n_dropped = len(obs) - len(shared_ids)
    if n_dropped > 0:
        log.warning(
            "  %d cells/spots in clean_data.h5ad not found in composition.csv "
            "— they will be excluded from dim 2.",
            n_dropped,
        )
        obs = obs.loc[shared_ids]
        comp_df = comp_df.loc[shared_ids]

    # ── Determine global cell-type vocabulary ─────────────────────────────────
    if is_proportion:
        vocab: list = list(comp_df.columns)
        log.info(
            "  Branch B: proportion matrix (%d spots × %d cell types)",
            len(comp_df), len(vocab),
        )
    else:
        vocab = sorted(comp_df.iloc[:, 0].unique().tolist())
        log.info(
            "  Branch A: %d cells | %d cell types in global vocabulary",
            len(comp_df), len(vocab),
        )

    log.info(
        "  Niches: %d | Samples: %d",
        obs["niche"].nunique(), obs["sample"].nunique(),
    )

    # ── Compute JSD per niche ─────────────────────────────────────────────────
    rows = []

    for niche in sorted(obs["niche"].unique()):
        niche_mask = (obs["niche"] == niche).values
        niche_obs = obs[niche_mask]
        samples_in_niche = niche_obs["sample"].unique()

        # Single-sample niches: stability undefined → N/A
        if len(samples_in_niche) < 2:
            rows.append(
                {
                    "niche": niche,
                    "n_samples": len(samples_in_niche),
                    "mean_JSD": np.nan,
                    "max_JSD": np.nan,
                    "min_JSD": np.nan,
                    "std_JSD": np.nan,
                    "stability_label": "N/A",
                    "reading": "Single sample — cross-sample stability is undefined; not 0.",
                }
            )
            continue

        # Per-sample composition vectors, aligned to GLOBAL vocabulary
        per_sample_vecs: dict = {}
        for sample in samples_in_niche:
            sample_mask = (niche_obs["sample"] == sample).values
            cell_ids = niche_obs.index[sample_mask]

            if is_proportion:
                # Branch B: mean of proportion rows
                vec = mean_proportion_vector(comp_df.loc[cell_ids, vocab].values)
            else:
                # Branch A: frequency distribution of cell-type labels
                labels = comp_df.loc[cell_ids].iloc[:, 0].values
                vec = freq_vector_from_labels(labels, vocab)

            per_sample_vecs[sample] = vec

        # Global centroid = equal-weight mean of per-sample vectors
        mat = np.vstack(list(per_sample_vecs.values()))   # (n_samples × n_types)
        centroid = mat.mean(axis=0)

        # JSD of each sample vs the centroid
        jsd_values = np.array(
            [js_divergence(v, centroid) for v in per_sample_vecs.values()]
        )

        mean_jsd = float(np.mean(jsd_values))
        stability_label = "high" if mean_jsd < args.jsd_threshold else "low"

        reading = (
            f"mean JSD = {mean_jsd:.3f} "
            f"({'high' if stability_label == 'high' else 'low'} stability — "
            f"composition {'is consistent' if stability_label == 'high' else 'varies'} "
            f"across {len(samples_in_niche)} samples)."
        )

        rows.append(
            {
                "niche": niche,
                "n_samples": len(samples_in_niche),
                "mean_JSD": round(mean_jsd, 4),
                "max_JSD": round(float(np.max(jsd_values)), 4),
                "min_JSD": round(float(np.min(jsd_values)), 4),
                "std_JSD": round(float(np.std(jsd_values)), 4),
                "stability_label": stability_label,
                "reading": reading,
            }
        )

    result = (
        pd.DataFrame(rows)
        .set_index("niche")
        .sort_values("mean_JSD", na_position="last")
    )

    out_path = outdir / "dim2_stability.csv"
    result.to_csv(out_path)
    log.info("Written: %s  (%d niches)", out_path, len(result))

    # ── Console summary ────────────────────────────────────────────────────────
    valid = result[result["mean_JSD"].notna()]
    log.info("=" * 60)
    log.info("Dimension 2 — Cross-sample Stability (JSD vs centroid)")
    log.info(
        "  High-stability niches  (mean JSD < %.2f): %d / %d",
        args.jsd_threshold,
        (result["stability_label"] == "high").sum(),
        len(result),
    )
    log.info(
        "  Low-stability niches   (mean JSD ≥ %.2f): %d",
        args.jsd_threshold,
        (result["stability_label"] == "low").sum(),
    )
    log.info(
        "  Single-sample niches   (N/A): %d",
        (result["stability_label"] == "N/A").sum(),
    )
    if len(valid) > 0:
        log.info(
            "  mean JSD range: [%.4f, %.4f]  |  median: %.4f",
            valid["mean_JSD"].min(),
            valid["mean_JSD"].max(),
            valid["mean_JSD"].median(),
        )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
