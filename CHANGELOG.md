# Changelog

## 2.0.0 — 2026-10-21

Compared with the version used in the [paper](https://www.mdpi.com/2072-4292/18/15/2529) (Soto Parra, Farò & Zolezzi 2026, *Remote Sensing* 18, 2529; tag `v1.0.0`).

- Released under the MIT License.

### Fixed
- Training blocks at polygon edges were padded with black pixels; blocks now sit on the map grid
  and are used only if at least 90 % lies inside the polygon (`min_label_coverage`).
- Class weights were applied twice.
- Map tiles could drop or shift blocks when the tile size was not a multiple of the block size.
- Cross-validation scores were optimistic (feature selection saw the test polygons); spatial CV
  is now nested.
- A missing `roi` layer silently used the seed points as ROI.
- Alpha bands and white (255,255,255) no-data borders are now recognised as no-data.
- Re-running the sampling step no longer erases annotations.

### Added
- Multi-scale texture features (GLCM at 1/2/4 px, LBP at radii 1–3, edge strength).
- Confidence map (probability of the mapped class) and optional class probabilities.
- Mosaic-edge blocks are classified from their neighbours.
- Validation report with user's/producer's accuracy and class areas (ha) with confidence
  intervals; a labelled sample can be reused to assess a re-trained map.
- Diagnostics (Step 9): prioritised tips to improve the model and suggested places to annotate.
- Seed points only on valid image pixels, optionally inside the ROI.
- Tests (`tests/`) and a benchmark against existing validation samples (`benchmarks/`).

### Changed
- Capped class weights (rare classes at most 10× the most frequent class) and a feature-selection
  rule that keeps fewer features.
- About 10× faster map classification (vectorised features, optional parallel processing with
  `n_jobs`) with lower memory use.
- Fewer dependencies (no fuzzywuzzy, seaborn; XGBoost and LightGBM optional).

### Note
- The maps in the paper were produced with 50 px blocks at all four sites.
