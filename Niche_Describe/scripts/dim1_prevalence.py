#!/usr/bin/env python3
"""
dim1_prevalence.py — Dimension 1: Cross-sample Prevalence
==========================================================
How broadly does each niche appear across samples?

Reads
-----
  clean_data.h5ad     (produced by validate.py)

Writes
------
  dim1_prevalence.csv

Metrics (per niche)
-------------------
  SPR              Sample Prevalence Rate = # samples containing the niche
                   / total # samples.
                   ≥ 0.20 → ubiquitous (default threshold, tunable)

  gini             Gini coefficient of the niche's NORMALIZED per-sample
                   proportions (niche cell count ÷ that sample's total cells),
                   computed across ALL samples (zeros included for absent samples).
                   0 = perfectly even; ~1 = maximally concentrated.
                   > 0.7 → highly concentrated (sample-dominant)

  top5pct_share    Fraction of the niche's total cells contributed by the
                   top-5% of samples (by niche proportion). < 0.30 → even.

CRITICAL: Gini and top5pct_share are computed on NORMALIZED per-sample
proportions, NOT raw cell counts (Pitfall 5).

Usage
-----
  python dim1_prevalence.py --indir results/ --outdir results/

Run after validate.py.
"""

import argparse
import logging
import sys
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
        description="Dimension 1 — Cross-sample prevalence (SPR / Gini / Top-5%).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--indir", default=".",
                   help="Directory containing clean_data.h5ad (default: .)")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: same as --indir)")
    p.add_argument("--spr-threshold", type=float, default=0.20,
                   help="SPR threshold for 'ubiquitous' flag (default: 0.20)")
    p.add_argument("--gini-threshold", type=float, default=0.70,
                   help="Gini threshold for 'concentrated' flag (default: 0.70)")
    p.add_argument("--top5-threshold", type=float, default=0.30,
                   help="Top-5%% concentration threshold for 'concentrated' flag (default: 0.30)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────
def gini_coefficient(values: np.ndarray) -> float:
    """
    Gini coefficient of a non-negative 1-D array (zeros included for absent
    samples).

    Formula (Lorenz-curve, rank-weighted):
        G = (2 · Σ_{k=1}^{n} k·x_k) / (n · Σx_k)  −  (n+1)/n
    where x_k are sorted ascending and k is the 1-based rank.

    Returns NaN when the sum of values is zero (niche has no cells).
    Range: [0, 1]; 0 = perfectly even; approaches 1 as concentration increases.
    """
    values = np.sort(np.asarray(values, dtype=float))
    total = values.sum()
    if total == 0.0:
        return np.nan
    n = len(values)
    ranks = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.dot(ranks, values)) / (n * total) - (n + 1.0) / n)


def top_k_pct_share(per_sample_raw_counts: pd.Series, top_pct: float = 0.05) -> float:
    """
    Fraction of the niche's total cells contributed by the top `top_pct`
    fraction of samples (ranked by raw count descending).

    At minimum, 1 sample is included in the 'top-k' group (ceiling).

    Note: we rank by raw counts but normalize the *result* to a proportion
    of the niche's global total — the numerator is raw counts (reflecting
    absolute contribution), the denominator is the niche's total cell count.
    This is the standard Top-5% concentration metric.
    """
    k = max(1, int(np.ceil(len(per_sample_raw_counts) * top_pct)))
    top_k_sum = per_sample_raw_counts.nlargest(k).sum()
    total = per_sample_raw_counts.sum()
    return float(top_k_sum / total) if total > 0 else np.nan


def prevalence_reading(spr: float, gini: float, top5: float,
                       spr_thresh: float, gini_thresh: float, top5_thresh: float) -> str:
    """Plain-language one-line reading of the three prevalence metrics."""
    parts = []
    if np.isnan(spr):
        return "insufficient data"
    parts.append(f"found in {spr * 100:.0f}% of samples")
    if not np.isnan(gini):
        if gini > gini_thresh:
            parts.append(f"concentrated (Gini={gini:.2f} > {gini_thresh})")
        else:
            parts.append(f"evenly distributed (Gini={gini:.2f})")
    if not np.isnan(top5):
        if top5 > top5_thresh:
            parts.append(f"top-5%% samples hold {top5 * 100:.0f}%% of cells")
        else:
            parts.append(f"top-5%% samples hold {top5 * 100:.0f}%% of cells (even)")
    return "; ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    indir = Path(args.indir)
    outdir = Path(args.outdir) if args.outdir else indir
    outdir.mkdir(parents=True, exist_ok=True)

    h5ad_path = indir / "clean_data.h5ad"
    if not h5ad_path.exists():
        log.error("clean_data.h5ad not found in %s. Run validate.py first.", indir)
        sys.exit(1)

    log.info("Loading %s", h5ad_path)
    adata = ad.read_h5ad(h5ad_path)
    obs = adata.obs[["niche", "sample"]].copy()

    all_samples = obs["sample"].unique()
    S = len(all_samples)
    all_niches = obs["niche"].unique()
    log.info(
        "  %d cells/spots | %d niches | %d samples",
        len(obs), len(all_niches), S,
    )

    # Per-sample total counts (denominator for normalized proportions)
    sample_totals = obs.groupby("sample", observed=True).size()

    rows = []
    for niche in sorted(all_niches):
        niche_group = obs[obs["niche"] == niche]

        # Raw counts per sample for this niche (only samples where it appears)
        per_sample_raw = niche_group.groupby("sample", observed=True).size()

        # Normalized proportion: niche cells / sample total cells
        # Fills 0.0 for samples where the niche is absent (required for Gini/Top-5%)
        per_sample_norm = (per_sample_raw / sample_totals).reindex(all_samples, fill_value=0.0)

        n_present = int((per_sample_norm > 0).sum())
        spr = n_present / S
        gini = gini_coefficient(per_sample_norm.values)
        top5 = top_k_pct_share(per_sample_raw, top_pct=0.05)

        rows.append(
            {
                "niche": niche,
                "n_cells_total": int(len(niche_group)),
                "n_samples_present": n_present,
                "n_samples_total": S,
                "SPR": round(spr, 4),
                "gini": round(gini, 4) if not np.isnan(gini) else np.nan,
                "top5pct_share": round(top5, 4) if not np.isnan(top5) else np.nan,
                "is_ubiquitous": spr >= args.spr_threshold,
                "is_sample_specific": n_present == 1,
                "is_concentrated_gini": (not np.isnan(gini)) and (gini > args.gini_threshold),
                "is_concentrated_top5": (not np.isnan(top5)) and (top5 > args.top5_threshold),
                "reading": prevalence_reading(
                    spr, gini, top5,
                    args.spr_threshold, args.gini_threshold, args.top5_threshold,
                ),
            }
        )

    result = (
        pd.DataFrame(rows)
        .set_index("niche")
        .sort_values("SPR", ascending=False)
    )

    out_path = outdir / "dim1_prevalence.csv"
    result.to_csv(out_path)
    log.info("Written: %s  (%d niches)", out_path, len(result))

    # ── Console summary ────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Dimension 1 — Cross-sample Prevalence")
    log.info(
        "  Ubiquitous niches      (SPR ≥ %.0f%%): %d / %d",
        args.spr_threshold * 100,
        result["is_ubiquitous"].sum(),
        len(result),
    )
    log.info(
        "  Sample-specific niches (present in 1 sample only): %d",
        result["is_sample_specific"].sum(),
    )
    log.info(
        "  Concentrated by Gini   (Gini > %.2f): %d",
        args.gini_threshold,
        result["is_concentrated_gini"].sum(),
    )
    log.info(
        "  Concentrated by Top-5%% (share > %.0f%%): %d",
        args.top5_threshold * 100,
        result["is_concentrated_top5"].sum(),
    )
    log.info("  Median SPR : %.3f", result["SPR"].median())
    log.info("  Median Gini: %.3f", result["gini"].median())
    log.info("=" * 60)


if __name__ == "__main__":
    main()
