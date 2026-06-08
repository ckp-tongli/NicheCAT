#!/usr/bin/env python3
"""
dim3_diversity.py — Dimension 3: Internal Composition Diversity
================================================================
How mixed vs. dominated is each niche internally?

Reads
-----
  clean_data.h5ad     (produced by validate.py)
  composition.csv     (produced by validate.py, Branch A or B only)

Writes
------
  dim3_diversity.csv
  dim3_diversity_per_sample.csv   (optional, with --per-sample)

Metrics (per niche, on the niche's GLOBAL composition)
-------
  simpson_D    1 − Σpᵢ²         (Gini-Simpson index)
               Range [0, 1]; higher = more diverse.
               Interpretation: probability that two randomly drawn cells
               from this niche have different cell types.
               > 0.70 → well-mixed; < 0.30 → dominated by one type.

  shannon_H    −Σpᵢ ln pᵢ       (Shannon entropy, natural-log / nats)
               Range [0, ln K]; higher = more even.
               More sensitive to rare cell types than Simpson.

  shannon_Hn   shannon_H / ln(K)  (normalised by log of richness K)
               Range [0, 1]; 1 = perfectly even across all observed types.
               NaN when K = 0.

  n_celltypes  Number of cell types with pᵢ > 0 in this niche (richness K).

All metrics are computed on PROPORTIONS, not raw cell counts (Pitfall 5).
On Branch C the script exits with code 2 and a clear message.

Usage
-----
  python dim3_diversity.py --indir results/ --outdir results/
  python dim3_diversity.py --indir results/ --per-sample   # adds per-sample detail

Run after validate.py.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

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
        description="Dimension 3 — Internal composition diversity (Simpson / Shannon).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--indir", default=".",
                   help="Directory containing clean_data.h5ad and composition.csv "
                        "(default: .)")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: same as --indir)")
    p.add_argument("--per-sample", action="store_true",
                   help="Also compute diversity within each sample×niche group "
                        "and write dim3_diversity_per_sample.csv")
    p.add_argument("--simpson-high", type=float, default=0.70,
                   help="Simpson D threshold for 'high diversity' flag (default: 0.70)")
    p.add_argument("--simpson-low", type=float, default=0.30,
                   help="Simpson D threshold for 'low diversity' flag (default: 0.30)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Diversity metrics (pure functions — independently testable)
# ──────────────────────────────────────────────────────────────────────────────
def simpson_diversity(proportions: np.ndarray) -> float:
    """
    Gini-Simpson diversity index: D = 1 − Σpᵢ²

    Parameters
    ----------
    proportions : 1-D array of non-negative values.
        Need not sum to 1 — normalised internally.

    Returns
    -------
    float in [0, 1]; higher → more diverse.
    NaN when the sum of proportions is zero (empty group).
    """
    p = np.asarray(proportions, dtype=float)
    p = p[p > 0]
    if p.sum() == 0:
        return np.nan
    p /= p.sum()
    return float(1.0 - np.dot(p, p))


def shannon_entropy(
    proportions: np.ndarray,
) -> tuple[float, float, int]:
    """
    Shannon entropy H = −Σpᵢ ln pᵢ (natural-log / nats).

    Parameters
    ----------
    proportions : 1-D array of non-negative values.

    Returns
    -------
    (H, H_normalised, K)
      H            : Shannon entropy in nats.
      H_normalised : H / ln(K); equals NaN when K ≤ 1.
      K            : Richness (number of types with pᵢ > 0).
    """
    p = np.asarray(proportions, dtype=float)
    p = p[p > 0]
    if p.sum() == 0:
        return np.nan, np.nan, 0
    p /= p.sum()
    K = len(p)
    H = float(-np.dot(p, np.log(p)))   # 0·log(0) → 0 handled by p[p>0] filter
    H_n = H / np.log(K) if K > 1 else (1.0 if K == 1 else np.nan)
    return H, H_n, K


def diversity_metrics(proportions: np.ndarray) -> dict:
    """Compute all three diversity metrics for one composition vector."""
    D = simpson_diversity(proportions)
    H, H_n, K = shannon_entropy(proportions)
    return {
        "simpson_D": round(D, 4) if not np.isnan(D) else np.nan,
        "shannon_H": round(H, 4) if not np.isnan(H) else np.nan,
        "shannon_Hn": round(H_n, 4) if not np.isnan(H_n) else np.nan,
        "n_celltypes": K,
    }


def diversity_reading(
    D: float, H: float, n_types: int, high_thresh: float, low_thresh: float
) -> str:
    """Plain-language one-line reading of the diversity metrics."""
    if np.isnan(D):
        return "insufficient data"
    label = (
        "well-mixed, no dominant type"
        if D > high_thresh
        else "dominated by a single type or few types"
        if D < low_thresh
        else "moderately mixed"
    )
    return (
        f"Simpson D = {D:.2f} → {label}; "
        f"Shannon H = {H:.2f} nats across {n_types} cell types."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Composition vector builders (both branches)
# ──────────────────────────────────────────────────────────────────────────────
def build_composition_vector(
    cell_ids: pd.Index,
    comp_df: pd.DataFrame,
    vocab: list,
    is_proportion: bool,
) -> np.ndarray:
    """
    Return a normalised composition vector aligned to `vocab`.

    Branch A (is_proportion=False): comp_df has one column 'cell_type';
        compute frequency distribution over vocab.
    Branch B (is_proportion=True): comp_df has one column per cell type;
        take the (unweighted) mean of rows — each row is a proportion vector.
    """
    if is_proportion:
        return comp_df.loc[cell_ids, vocab].values.mean(axis=0).astype(float)
    else:
        label_counts = comp_df.loc[cell_ids].iloc[:, 0].value_counts()
        vec = np.array([label_counts.get(ct, 0) for ct in vocab], dtype=float)
        total = vec.sum()
        return vec / total if total > 0 else vec


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

    # ── Guards ────────────────────────────────────────────────────────────────
    if not h5ad_path.exists():
        log.error(
            "clean_data.h5ad not found in %s. Run validate.py first.", indir
        )
        sys.exit(1)

    if not comp_path.exists():
        log.error(
            "composition.csv not found in %s.\n"
            "This is Branch C (no cell-type annotation or proportion matrix).\n"
            "Dimension 3 requires a composition axis to measure internal diversity.\n"
            "Action: supply --celltype (high-res) or --proportion (low-res) to "
            "validate.py and re-run.",
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

    # Restrict to shared IDs
    shared_ids = obs.index.intersection(comp_df.index)
    n_dropped = len(obs) - len(shared_ids)
    if n_dropped > 0:
        log.warning(
            "  %d cells/spots in clean_data.h5ad not found in composition.csv — excluded.",
            n_dropped,
        )
        obs = obs.loc[shared_ids]
        comp_df = comp_df.loc[shared_ids]

    # Global vocabulary (all cell types seen across the entire dataset)
    if is_proportion:
        vocab: list = list(comp_df.columns)
        log.info(
            "  Branch B: %d spots × %d cell types", len(comp_df), len(vocab)
        )
    else:
        vocab = sorted(comp_df.iloc[:, 0].unique().tolist())
        log.info(
            "  Branch A: %d cells | %d cell types", len(comp_df), len(vocab)
        )

    log.info(
        "  Niches: %d | Samples: %d",
        obs["niche"].nunique(), obs["sample"].nunique(),
    )

    # ── Global diversity per niche ─────────────────────────────────────────────
    global_rows = []
    per_sample_rows: list[dict] = []

    for niche in sorted(obs["niche"].unique()):
        niche_mask = obs["niche"] == niche
        niche_ids = obs.index[niche_mask.values]

        # Global composition (all cells in this niche across all samples)
        global_props = build_composition_vector(niche_ids, comp_df, vocab, is_proportion)
        metrics = diversity_metrics(global_props)

        global_rows.append(
            {
                "niche": niche,
                "n_cells_total": len(niche_ids),
                **metrics,
                "diversity_label": (
                    "high" if (not np.isnan(metrics["simpson_D"]) and metrics["simpson_D"] > args.simpson_high)
                    else "low" if (not np.isnan(metrics["simpson_D"]) and metrics["simpson_D"] < args.simpson_low)
                    else "medium"
                ),
                "reading": diversity_reading(
                    metrics["simpson_D"],
                    metrics["shannon_H"],
                    metrics["n_celltypes"],
                    args.simpson_high,
                    args.simpson_low,
                ),
            }
        )

        # Optional per-sample breakdown
        if args.per_sample:
            niche_obs = obs[niche_mask]
            for sample in sorted(niche_obs["sample"].unique()):
                sample_ids = niche_obs.index[niche_obs["sample"] == sample]
                s_props = build_composition_vector(sample_ids, comp_df, vocab, is_proportion)
                s_metrics = diversity_metrics(s_props)
                per_sample_rows.append(
                    {
                        "niche": niche,
                        "sample": sample,
                        "n_cells": len(sample_ids),
                        **s_metrics,
                    }
                )

    # ── Write global diversity CSV ─────────────────────────────────────────────
    global_result = (
        pd.DataFrame(global_rows)
        .set_index("niche")
        .sort_values("simpson_D", ascending=False)
    )

    out_path = outdir / "dim3_diversity.csv"
    global_result.to_csv(out_path)
    log.info("Written: %s  (%d niches)", out_path, len(global_result))

    # ── Write optional per-sample diversity CSV ────────────────────────────────
    if args.per_sample and per_sample_rows:
        ps_result = pd.DataFrame(per_sample_rows)
        ps_out = outdir / "dim3_diversity_per_sample.csv"
        ps_result.to_csv(ps_out, index=False)
        log.info("Written: %s  (%d rows, per-sample detail)", ps_out, len(ps_result))

    # ── Console summary ────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Dimension 3 — Internal Composition Diversity")
    log.info(
        "  High-diversity niches  (Simpson D > %.2f): %d / %d",
        args.simpson_high,
        (global_result["diversity_label"] == "high").sum(),
        len(global_result),
    )
    log.info(
        "  Medium-diversity niches:                   %d",
        (global_result["diversity_label"] == "medium").sum(),
    )
    log.info(
        "  Low-diversity niches   (Simpson D < %.2f): %d",
        args.simpson_low,
        (global_result["diversity_label"] == "low").sum(),
    )
    log.info(
        "  Median Simpson D : %.4f", global_result["simpson_D"].median()
    )
    log.info(
        "  Median Shannon H : %.4f nats", global_result["shannon_H"].median()
    )
    log.info(
        "  Median norm. Shannon Hn: %.4f", global_result["shannon_Hn"].median()
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
