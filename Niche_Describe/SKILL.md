---
name: niche-description
description: Quantitatively describe (characterize) pre-clustered cellular niches / cellular neighborhoods from spatial omics data. Use this whenever a user has ALREADY assigned niche / neighborhood / spatial-domain labels (e.g. from Schürch-style kNN clustering, BANKSY, CytoCommunity, GASTON, Seurat, etc.) and now wants a standardized QC / description report covering how broadly each niche appears across samples, how stable its composition is across samples, how diverse its internal cell-type makeup is, and whether it is spatially clustered within the tissue. Trigger this even when the user only says things like "describe my niches", "QC my neighborhoods", "are my niches reproducible / sample-specific", "how pure is each niche", "is niche X spatially clustered", or hands over spatial data + coordinates + niche labels + sample IDs — even if they don't use the word "niche" explicitly. High-resolution platforms (CosMx, Xenium, MERFISH) supply a cell×gene matrix plus a cell-type annotation column; low-resolution platforms (Visium, SlideSeq) supply a spot×gene matrix plus a spot×cell deconvolution proportion matrix. This skill does NOT discover or cluster niches (that is upstream) and does NOT interpret biological function (that is downstream); it produces the quantitative description that feeds those steps.
---

# Niche Description (single-niche characterization)

## What this skill is — and is NOT

This skill takes niches that have **already been clustered upstream** and produces a **standardized quantitative description** of each one. The niche labels are an **input**, not something this skill computes.

**In scope** — four description dimensions, each computed **per niche**:

1. **Cross-sample prevalence** — how broadly the niche appears across samples (SPR / Gini / Top-5%)
2. **Cross-sample stability** — how consistent the niche's cell-type composition is across samples (Jensen–Shannon Divergence)
3. **Internal composition diversity** — how mixed vs. dominated the niche is internally (Simpson / Shannon)
4. **Single-niche spatial clustering** — whether the niche forms spatial clumps within the tissue (binarized Join Count statistic)

**Explicitly OUT of scope** (do not drift into these):

- **Niche discovery / re-clustering.** If the user has no labels yet, stop and tell them this skill needs labels; point them to upstream tools (Schürch kNN+k-means, BANKSY, CytoCommunity, GASTON…). Do not "helpfully" re-cluster.
- **Inter-niche spatial relationships** (neighborhood enrichment, co-occurrence, "is niche A next to niche B"). This is **single-niche** analysis only. Do not compute or report cross-niche colocalization.
- **Biological / functional interpretation.** Do not guess what each niche "does" biologically. Output the numbers and one-line plain-language readings of the *statistics*; leave functional meaning to the user / downstream. State this boundary in the report.

If a request is for discovery, inter-niche relationships, or functional annotation, say so plainly and offer what is in scope instead.

---

## Required inputs

The required files differ by platform resolution. All files are joined on a **shared cell/spot ID** — never assume row order (see Pitfall 2).

### High-resolution platforms (CosMx, Xenium, MERFISH, Stereo-seq at cell level, …) — **five files**

| # | File | Contents |
|---|---|---|
| 1 | **cell×gene matrix** | Expression counts, one row per cell |
| 2 | **cell-type annotation** | A single column of cell-type labels, one entry per cell (e.g. an `.obs` column `"cell_type"`). This is the composition axis for dims 2–3. |
| 3 | **coordinates** | x, y per cell |
| 4 | **niche labels** | One cluster label per cell (single column) |
| 5 | **sample IDs** | Which sample/slice each cell belongs to |

> The cell×gene matrix (file 1) is used for validation, QC, and any expression-level diagnostics. Cell-type composition for dims 2–3 is derived **directly from the cell-type annotation column** (file 2), not from the expression matrix. If a cell-type annotation column is absent, dims 2–3 cannot run — see Branch C below.

### Low-resolution platforms (Visium, Visium HD, SlideSeq, …) — **five files**

| # | File | Contents |
|---|---|---|
| 1 | **spot×gene matrix** | Expression counts, one row per spot |
| 2 | **spot×cell proportion matrix** | Deconvolution output — one row per spot, one column per cell type, each row sums to ≈ 1. This is the composition axis for dims 2–3. |
| 3 | **coordinates** | x, y per spot |
| 4 | **niche labels** | One cluster label per spot (single column) |
| 5 | **sample IDs** | Which sample/slice each spot belongs to |

> Do **not** argmax the spot×cell matrix to a single "dominant" cell type before computing dims 2–3 — that discards the proportional information you were given.

**Summary of the composition axis by platform:**

| Platform | Composition axis for dims 2–3 |
|---|---|
| High-resolution | Cell-type annotation column (one label per cell) |
| Low-resolution | spot×cell proportion matrix (continuous proportions per spot) |

---

## STEP 0 — Ask before you compute (elicitation)

The single biggest failure mode in spatial omics is **running on defaults and producing numbers that look reasonable but are wrong**. Before any computation, resolve the questions below. Use the interactive option-button tool where available; ask at most ~3 at a time; infer from the data/conversation when you safely can and state the inferred value rather than asking.

Do **not** skip this step because the user "just wants results fast." A wrong coordinate unit or a cross-sample graph silently corrupts the entire report.

| Must clarify | Why it matters | Notes |
|---|---|---|
| **Resolution: single-cell vs spot-based?** | Picks the branch below; determines which file provides the composition axis | e.g. CosMx/Xenium/MERFISH = high-resolution; Visium/Visium HD = low-resolution |
| **(high-res) Is a cell-type annotation column present?** | Required for dims 2–3; its absence forces Branch C | e.g. `.obs["cell_type"]`. Ask where it lives if not obvious |
| **(low-res) Is a spot×cell proportion matrix present?** | Required for dims 2–3; absence forces Branch C | Must have one column per cell type, rows summing to ≈ 1 |
| **Coordinate unit: physical (µm) or pixels?** | The Join Count graph is built on physical distance; wrong unit ⇒ wrong neighbors | Must confirm; do not guess. See `references/platform_notes.md` |
| **Spatial graph definition: kNN (k=?), fixed radius (r=? µm), or Delaunay?** | Directly sets the **spatial scale** of the clustering test | Regular spot arrays → grid/6-neighbor; imaging single-cell → radius or kNN. squidpy's 6-NN default is not always right |
| **Multi-sample / multi-slice?** | Determines per-sample handling | Slices have independent coordinate systems — see Pitfall 1 |
| **Minimum niche size** | Tiny niche×sample groups give noise | Default: drop any niche×sample group with < **20** cells/spots (tunable) |

---

## STEP 1 — Branch by platform

The four dimensions are the same; only how the **composition vector** (the cell-type-proportion vector used by dims 2 and 3) is built differs.

**Branch A — High-resolution platform (CosMx, Xenium, MERFISH, Stereo-seq at cell level, …)**
The composition axis comes from the **cell-type annotation column** (one label per cell, e.g. `obs["cell_type"]`). Composition of a niche (globally or within a sample) = the frequency distribution over cell types of the cells assigned to that niche. The cell×gene matrix is used for QC and validation but is **not** the source of composition.

**Branch B — Low-resolution platform WITH a spot×cell proportion matrix (Visium, SlideSeq, …)**
Each spot carries a deconvolution-derived cell-type proportion vector. A niche's composition = the (size-weighted) mean proportion vector over its member spots. Do **not** argmax spots to a single dominant type before averaging — that discards the proportional information you were given. The spot×gene matrix is used for QC and validation but composition comes entirely from the proportion matrix.

**Branch C — Low-resolution platform WITHOUT a proportion matrix (spot×gene only, no deconvolution)**
There is no cell-type axis yet. You **cannot** compute dims 2–3 honestly without one. Options, in order of preference — confirm with the user:
1. They supply or point to a deconvolution result (→ becomes Branch B), or
2. They accept a clearly-labeled proxy composition (e.g. over marker-gene signature scores or a coarse expression clustering), reported as a proxy, not as cell types.
Until a cell-type (or explicitly-labeled proxy) axis exists, run only dims 1 and 4 and say why 2–3 are deferred.

> **Note:** A high-resolution platform without a cell-type annotation column is effectively the same situation as Branch C — run dims 1 and 4 only and ask the user to supply annotation.

Dimensions 1 (prevalence) and 4 (spatial clustering) are **platform-agnostic** — they need only labels, sample IDs, and coordinates.

---

## STEP 2 — The three ironclad data-prep rules

These three are the highest-frequency, highest-damage agent errors. Treat them as non-negotiable.

### RULE 1 — BUILD THE SPATIAL GRAPH PER SAMPLE. NEVER ACROSS SAMPLES.

Every slice/sample has its **own coordinate system**. Stacking multiple samples' (x, y) into one graph creates fake adjacencies between cells that are physically in different tissues. **All spatial computation (the Join Count test in dim 4) must be run per-sample and then aggregated.** Never connect edges across sample boundaries.

### RULE 2 — JOIN THE FIVE FILES BY ID. NEVER ASSUME ROW ORDER.

All five input files must be aligned on a **shared cell/spot ID**, not concatenated by position. A naive `pd.concat` on mismatched orders silently misassigns every label. The loader must inner-join on ID and **report how many rows were dropped** for non-matching.

### RULE 3 — NICHE LABELS ARE CATEGORICAL. NEVER TREAT THEM AS CONTINUOUS.

Niche labels are unordered categories ("niche 3" is not greater than "niche 1"). **Do not feed raw niche IDs into Moran's I, Geary's C, or Getis-Ord G as if they were continuous values** — the result is meaningless. To ask "is *this* niche spatially clustered?", **binarize**: code membership of the target niche as 1 and everything else as 0, then apply the **Join Count statistic** (the correct test for a binary spatial variable). Dimension 4 does exactly this, one niche at a time, per sample.

---

## STEP 3 — Other prep, in `validate_and_load.py`

The loader (`scripts/validate_and_load.py`) is the most important script and must run first. It produces one clean AnnData (expression matrix in `.X`, coordinates in `.obsm['spatial']`, cell-type annotation and niche label and sample ID in `.obs`) that every downstream script consumes. It must:

- Inner-join **all five files** on ID; report dropped-row counts for each join (Rule 2).
  - For high-resolution platforms: join expression matrix, cell-type annotation column, coordinates, niche labels, and sample IDs.
  - For low-resolution platforms: join spot×gene matrix, spot×cell proportion matrix, coordinates, niche labels, and sample IDs. Store the proportion matrix in `.obsm['cell_proportions']`.
- **Drop "unassigned" niche labels** (`-1`, `NA`, `"unassigned"`, `""`, etc.) — these are not a real niche. Confirm the sentinel value with the user if ambiguous.
- Clean coordinates: drop/flag NaNs, exact duplicate coordinates, and obvious outliers that would distort graph construction.
- Apply the **minimum niche size** filter (default 20 per niche×sample group); report what was filtered.
- Normalize composition to proportions *per group* before any prevalence/diversity math (Pitfall 5).
  - High-res: compute per-group cell-type frequencies from the annotation column.
  - Low-res: use the proportion matrix directly; verify rows sum to ≈ 1 (warn if any row deviates by > 0.01).
- **Fix a random seed** and record it (Pitfall 9).
- Write a `run_config.json` capturing every parameter, the seed, package versions, platform branch, and all drop/filter counts.

---

## STEP 4 — Compute the four dimensions

Run each as its own script so results are reusable and independently checkable. Every script reads the clean AnnData and writes a tidy CSV.

### Dimension 1 — Cross-sample prevalence  (`prevalence.py`)
How broadly does the niche appear across samples?

| Metric | Definition | Default reading threshold (tunable) |
|---|---|---|
| **Sample Prevalence Rate (SPR)** | # samples containing the niche / total samples | ≥ 20% (general) · ≥ 5% (rare-subtype studies) |
| **Gini coefficient** | Gini of the niche's **per-sample proportions** (its cell count ÷ that sample's total cells), across samples | < 0.5 even · > 0.7 concentrated |
| **Top-5% concentration** | Share of the niche's cells contributed by the top-5% samples | < 30% even |

**Critical:** Gini and Top-5% are computed on **normalized per-sample proportions**, NOT raw cell counts. Otherwise large samples trivially dominate and a ubiquitous niche looks falsely "concentrated" (Pitfall 5).

### Dimension 2 — Cross-sample stability  (`stability.py`)
Is the niche's cell-type composition consistent from sample to sample?

- **Jensen–Shannon Divergence (JSD):** for each niche, build its cell-type frequency vector within each sample, then measure dispersion. **Recommended:** each sample's vector vs. the niche's **global centroid** vector (robust, O(n)), rather than all-pairs (O(n²)). Lower = more consistent; default reading JSD < 0.3 = high stability.
- **Mandatory alignment:** every per-sample composition vector must be aligned to the **same global cell-type vocabulary**, with missing types filled as 0, before computing JSD — otherwise the vectors have different lengths and the divergence is invalid (Pitfall: composition vectors not aligned).
- Only meaningful for niches passing the prevalence/size filters; a niche present in 1 sample has no cross-sample stability — report it as N/A, not 0.

### Dimension 3 — Internal composition diversity  (`composition_diversity.py`)
How mixed vs. dominated is the niche internally? Computed per niche on its global composition (and optionally per sample).

- **Simpson diversity** D = 1 − Σpᵢ² (higher = more even / diverse; reflects probability two random cells differ).
- **Shannon entropy** H = −Σpᵢ ln pᵢ (more sensitive to rare types) — report alongside Simpson.
- Optionally normalized Simpson (1/D or Gini-Simpson) for cross-niche comparison.
- This is **cell-type-composition** diversity. On Branch C with only a proxy axis, label it clearly as proxy diversity, not cell-type diversity.

### Dimension 4 — Single-niche spatial clustering  (`spatial_clustering.py`)
Is the niche spatially clumped within tissue, vs. scattered?

- For each niche, **one at a time**: binarize (target niche = 1, else = 0).
- Build the spatial graph **per sample** (Rule 1) using the user-chosen kNN/radius/Delaunay (shared helper `build_graph.py`).
- Compute the **Join Count statistic** (observed same-label "1–1" joins vs. expected under spatial randomness), via `esda`/`libpysal`, with a permutation test (fixed seed). Positive/excess 1–1 joins ⇒ spatial clustering.
- Aggregate the per-sample results to a per-niche summary (e.g. count of samples where the niche is significantly clustered, and a combined/median statistic). **Do not pool cells across samples into one graph.**

After all four dimensions: **BH/FDR-correct** every p-value across the full niche × test grid (`fdr_correct.py`) before calling anything significant (Pitfall 10).

---

## STEP 5 — Assemble the report  (`assemble_report.py`)

Two outputs:

1. **Master table** (CSV): one row per niche, columns = every metric above (SPR, Gini, Top-5%, JSD, Simpson, Shannon, Join-Count stat + FDR-corrected p + #samples-clustered). This is the clean machine-readable handoff to downstream analysis.
2. **Markdown report** for humans, using **exactly** this structure:

```
# Niche Description Report
## Data overview            (samples, cells/spots, # niches, what was filtered/dropped)
## Input parameters         (platform, branch, coord unit, graph definition, all thresholds, random seed, package versions)   ← REQUIRED, reproducibility
## Dimension 1 — Cross-sample prevalence      (table + which niches are sample-specific)
## Dimension 2 — Cross-sample stability        (table + most-consistent / most-variable niches; N/A where single-sample)
## Dimension 3 — Internal composition diversity (table + high- vs low-diversity niches)
## Dimension 4 — Single-niche spatial clustering (Join-Count results per niche + FDR-corrected significance)
## Methods & limitations    (binarization for categorical labels, per-sample graphs, FDR correction, any proxy axis, single-niche scope) ← REQUIRED
```

The **Input parameters** and **Methods & limitations** sections are mandatory — they are the floor for scientific reproducibility and prevent silently-changed parameters. Each metric in the report gets **one plain-language sentence** explaining the statistic (the report's reader may not be a statistician), e.g. "Simpson D = 0.82 → this niche is internally well-mixed with no single dominant cell type." Explain the *number*, not the biology.

---

## Pitfalls checklist (verify against this before reporting)

1. **Cross-sample graph** — graphs built per sample, never pooled (Rule 1). ⚠️ most damaging
2. **Row-order join** — five files joined by ID with drop report, never `concat` by position (Rule 2).
3. **Categorical-as-continuous** — niche labels binarized for dim 4, never fed to Moran's I/Geary's C/G (Rule 3).
4. **No min-size filter** — tiny niche×sample groups excluded before stability/diversity.
5. **Un-normalized counts** — Gini/Top-5%/diversity on proportions, not raw counts.
6. **Dirty coordinates** — NaN / duplicate / outlier coordinates handled before graph build.
7. **(Branch C) no composition axis** — don't compute dims 2–3 off raw genes without a cell-type annotation (high-res) or proportion matrix (low-res); use a clearly-labeled proxy only if the user agrees, and label it as such throughout the report.
8. **Unassigned label** — `-1` / `NA` / `"unassigned"` excluded, not described as a niche.
9. **No seed** — random seed fixed and recorded; permutation tests reproducible.
10. **No multiple-testing correction** — BH/FDR applied across the niche × test grid before claiming significance.

---

## Scripts and references

`scripts/` — run in this order; each reads `clean_data.h5ad` and writes one CSV:

| Script | Reads | Writes | Notes |
|---|---|---|---|
| `validate.py` | 5 raw files | `clean_data.h5ad`, `composition.csv`, `run_config.json` | **Run first.** All others depend on this. High-res: cell-type annotation column → `composition.csv`. Low-res: spot×cell proportion matrix → stored in `.obsm['cell_proportions']` and also exported as `composition.csv`. |
| `dim1_prevalence.py` | `clean_data.h5ad` | `dim1_prevalence.csv` | Platform-agnostic |
| `dim2_stability.py` | `clean_data.h5ad` + `composition.csv` | `dim2_stability.csv` | Branch A/B only; exits clearly if `composition.csv` absent |
| `dim3_diversity.py` | `clean_data.h5ad` + `composition.csv` | `dim3_diversity.csv` | Branch A/B only; exits clearly if `composition.csv` absent |
| `dim4_spatial_clustering.py` | `clean_data.h5ad` | `dim4_spatial_clustering.csv` + `dim4_spatial_clustering_detail.csv` | Graph building + Join Count + FDR all in one script; platform-agnostic |

After all four dimension scripts run, the agent reads the four CSVs directly to assemble the report — no separate assembly script is needed.

**Merged from original 8-script design:**
- `build_graph.py` + `spatial_clustering.py` + `fdr_correct.py` → all inside `dim4_spatial_clustering.py`
- `assemble_report.py` → dropped; agent does inline assembly from the four CSVs


## Dependencies

`anndata`, `scipy`, `numpy`, `pandas`, `statsmodels` (BH/FDR). No squidpy, esda, libpysal, or scikit-bio needed — spatial graph and Join Count are implemented directly in `dim4_spatial_clustering.py` using `scipy.spatial.cKDTree`. Install with `pip install <pkg> --break-system-packages`.
