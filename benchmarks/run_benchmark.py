"""
Benchmark the pipeline on the study rivers and compare against the published maps.

For each river:
  1. Extract training features from the annotation polygons.
  2. Train with nested spatial CV (+ standard CV for the optimism gap).
  3. Re-use the existing, already-labelled design-based validation sample of the
     old map: the new model is evaluated on exactly the same blocks. Because the
     sample was stratified by the *old* map classes, accuracy of the new map is
     estimated with the stratified estimators of Stehman (2014, Int. J. Remote
     Sens. 35:4923-4939) for strata that differ from the map classes. A paired
     difference (new - old) with its standard error is also reported.

Usage (from the repository root):
    python -m benchmarks.run_benchmark --data-root data/rivers --out outputs/benchmark

<data-root>/<River>/ must hold orthophoto.tif, sampling_data.gpkg (annotations) and
validation_sample.gpkg (labelled validation sample of the published map); the rivers and
their annotation layers are listed in RIVERS.
"""
import argparse
import json
import os
import time

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import norm

from src.block_features import grid_transform
from src.features import run_extraction
from src.inference import predict_proba_blocks
from src.postprocess import finalize, NODATA
from src.validation import design_based_estimates, _stratified_mean
from src.modeling import train_pipeline

RIVERS = {
    'Aurino': {'layer': 'annotations'},
    'Brenta': {'layer': 'labels-rand'},
    'Piave': {'layer': 'annotations'},
    'Sarca': {'layer': 'annotations'},
}
# Published values (Table 5 spatial CV F1/OA, Table 6 design-based OA)
PAPER = {
    'Aurino': {'spatial_f1': 0.59, 'spatial_oa': 0.80, 'design_oa': 0.8474},
    'Piave': {'spatial_f1': 0.64, 'spatial_oa': 0.82, 'design_oa': 0.8053},
    'Brenta': {'spatial_f1': 0.59, 'spatial_oa': 0.64, 'design_oa': 0.6977},
    'Sarca': {'spatial_f1': 0.61, 'spatial_oa': 0.84, 'design_oa': 0.8785},
}


def _summary(estimate, se, z):
    return {'estimate': estimate, 'se': se, 'ci_low': estimate - z * se, 'ci_high': estimate + z * se}


def evaluate_on_old_sample(sample, new_pred, new_names, old_names, z):
    """Design-based accuracy of the new and the old map on the old validation sample."""
    stratum = sample['map_prediction'].astype(int).to_numpy()
    W = sample['stratum_weight'].to_numpy()
    ref = [old_names[int(v)] for v in sample['true_label']]
    old = [old_names[int(v)] for v in sample['map_prediction']]
    new = [new_names.get(int(v), 'nodata') for v in new_pred]
    classes = sorted(set(new_names.values()) | set(old_names.values()))
    est_new = design_based_estimates(stratum, W, new, ref, classes)
    est_old = design_based_estimates(stratum, W, old, ref, classes)

    # Paired difference of the two maps on the same units
    df = pd.DataFrame({'stratum': stratum, 'W': W,
                       'diff': (np.array(new) == np.array(ref)).astype(float) - (np.array(old) == np.array(ref))})
    diff, diff_se = _stratified_mean(df, 'diff')

    pn, po = est_new['per_class'].set_index('class'), est_old['per_class'].set_index('class')
    return {
        'n_points': len(df),
        'n_nodata_new': int(sum(v == 'nodata' for v in new)),
        'oa_new': _summary(est_new['overall']['accuracy'], est_new['overall']['se'], z),
        'oa_old': _summary(est_old['overall']['accuracy'], est_old['overall']['se'], z),
        'oa_diff': _summary(diff, diff_se, z),
        'per_class': [{'class': c, 'ua_new': pn.loc[c, 'user_accuracy'], 'ua_old': po.loc[c, 'user_accuracy'],
                       'pa_new': pn.loc[c, 'producer_accuracy'], 'pa_old': po.loc[c, 'producer_accuracy']}
                      for c in classes],
    }


def point_windows(ortho, model_dir, block_size, points, eval_block_size=None, halo=2):
    """
    Block probabilities around every validation point.

    The reference unit is the model block containing the point or, with `eval_block_size` (px),
    the eval_block_size square of the validation grid containing it. Each entry holds the class
    probabilities and valid fractions of the blocks overlapping that unit plus `halo` blocks on
    each side (so smoothing and edge filling see their neighbours), and the pixel overlap of
    every block with the reference unit.
    """
    model = joblib.load(os.path.join(model_dir, 'final_model.joblib'))
    feats = joblib.load(os.path.join(model_dir, 'final_model_features.joblib'))
    n_classes = len(joblib.load(os.path.join(model_dir, 'final_class_mapping.joblib')))
    unit = eval_block_size or block_size
    windows = []
    with rasterio.open(ortho) as src:
        pts = points.to_crs(src.crs)
        n_rows, n_cols = src.height // block_size, src.width // block_size
        for geom in pts.geometry:
            R, C = rasterio.transform.rowcol(grid_transform(src, unit), geom.x, geom.y)
            y0, y1, x0, x1 = R * unit, (R + 1) * unit, C * unit, (C + 1) * unit
            r0 = max(y0 // block_size - halo, 0)
            r1 = min((y1 - 1) // block_size + 1 + halo, n_rows)
            c0 = max(x0 // block_size - halo, 0)
            c1 = min((x1 - 1) // block_size + 1 + halo, n_cols)
            P, frac = predict_proba_blocks(src, model, feats, n_classes, block_size, r0, c0, r1 - r0, c1 - c0)
            by = np.arange(r0, r1) * block_size
            bx = np.arange(c0, c1) * block_size
            oy = np.clip(np.minimum(y1, by + block_size) - np.maximum(y0, by), 0, None)
            ox = np.clip(np.minimum(x1, bx + block_size) - np.maximum(x0, bx), 0, None)
            windows.append((P, frac, np.outer(oy, ox)))
    return windows


def labels_at_points(windows, smoothing=1, fill_min_valid=None):
    """Area-weighted majority of the post-processed block labels inside each reference unit."""
    preds = []
    for P, frac, overlap in windows:
        labels, _ = finalize(P, frac, smoothing, fill_min_valid)
        ok = (overlap > 0) & (labels != NODATA)
        if not ok.any():
            preds.append(NODATA)
            continue
        votes = np.bincount(labels[ok], weights=overlap[ok])
        preds.append(int(votes.argmax()))
    return np.array(preds)


def predict_points(ortho, model_dir, block_size, points, eval_block_size=None, smoothing=1, fill_min_valid=None):
    windows = point_windows(ortho, model_dir, block_size, points, eval_block_size, halo=max(smoothing // 2, 1))
    return labels_at_points(windows, smoothing, fill_min_valid)


def load_validation(name, data_root):
    river_dir = os.path.join(data_root, name)
    sample = gpd.read_file(os.path.join(river_dir, 'validation_sample.gpkg'))
    labels = os.path.join(river_dir, 'sampling_data.gpkg')
    old_classes = sorted(gpd.read_file(labels, layer=RIVERS[name]['layer'])['substrate'].dropna().unique())
    return sample, dict(enumerate(old_classes))     # old maps used the same alphabetical LabelEncoder order


def evaluate_river(name, data_root, out, block_size, z, eval_block_size=None, smoothing=1, fill_min_valid=None):
    sample, old_names = load_validation(name, data_root)
    new_names = joblib.load(os.path.join(out, 'models', 'final_class_mapping.joblib'))
    new_pred = predict_points(os.path.join(data_root, name, 'orthophoto.tif'), os.path.join(out, 'models'),
                              block_size, sample, eval_block_size, smoothing, fill_min_valid)
    design = evaluate_on_old_sample(sample, new_pred, new_names, old_names, z)
    design.update(eval_block_size=eval_block_size or block_size, smoothing=smoothing, fill_min_valid=fill_min_valid)
    return design


def run_river(name, data_root, out_root, block_size, coverage, nested, z, reuse_features, eval_block_size=None,
              smoothing=1, fill_min_valid=None):
    river_dir = os.path.join(data_root, name)
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    ortho = os.path.join(river_dir, 'orthophoto.tif')
    labels = os.path.join(river_dir, 'sampling_data.gpkg')
    csv = os.path.join(out, 'block_features.csv')
    timing = {}

    t = time.time()
    if not (reuse_features and os.path.exists(csv)):
        run_extraction(ortho, labels, RIVERS[name]['layer'], csv, block_size=block_size,
                       min_label_coverage=coverage, show_plot=False)
    timing['extraction_s'] = time.time() - t
    blocks = pd.read_csv(csv)

    t = time.time()
    _, feats, cv = train_pipeline(csv, os.path.join(out, 'models'), nested_cv=nested, random_cv=True,
                                  show_plots=False)
    timing['training_s'] = time.time() - t

    t = time.time()
    design = evaluate_river(name, data_root, out, block_size, z, eval_block_size, smoothing, fill_min_valid)
    timing['validation_s'] = time.time() - t

    return {
        'river': name,
        'block_size': block_size,
        'min_label_coverage': coverage,
        'blocks_per_class': blocks['class'].value_counts().to_dict(),
        'n_blocks': len(blocks),
        'cv': {k: v for k, v in cv.items() if k in ('n_samples', 'n_polygons', 'n_splits', 'selected_features')},
        'spatial_cv': {k: cv['spatial_cv'][k] for k in ('nested', 'accuracy', 'macro_f1', 'weighted_f1')},
        'standard_cv': {k: cv['standard_cv'][k] for k in ('accuracy', 'macro_f1', 'weighted_f1')},
        'design_based': design,
        'paper': PAPER.get(name),
        'timing': timing,
    }


def write_summary(results, path):
    lines = ["| River | Block (px) | Eval square (px) | Blocks | Spatial CV OA / macro-F1 (paper) | Spatial CV OA / macro-F1 (new) | "
             "Standard CV macro-F1 (new) | Design OA old map | Design OA new model | Δ (95% CI) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        d, p = r['design_based'], r['paper']
        lines.append(
            f"| {r['river']} | {r['block_size']} | {d.get('eval_block_size', r['block_size'])} | {r['n_blocks']} | {p['spatial_oa']:.2f} / {p['spatial_f1']:.2f} | "
            f"{r['spatial_cv']['accuracy']:.2f} / {r['spatial_cv']['macro_f1']:.2f} | "
            f"{r['standard_cv']['macro_f1']:.2f} | "
            f"{d['oa_old']['estimate']:.1%} ± {1.96 * d['oa_old']['se']:.1%} | "
            f"{d['oa_new']['estimate']:.1%} ± {1.96 * d['oa_new']['se']:.1%} | "
            f"{d['oa_diff']['estimate']:+.1%} ({d['oa_diff']['ci_low']:+.1%}, {d['oa_diff']['ci_high']:+.1%}) |")
    lines.append("")
    for r in results:
        lines.append(f"### {r['river']}  (blocks/class: {r['blocks_per_class']})")
        lines.append("| Class | UA old | UA new | PA old | PA new |")
        lines.append("|---|---|---|---|---|")
        for c in r['design_based']['per_class']:
            lines.append(f"| {c['class']} | {c['ua_old']:.2f} | {c['ua_new']:.2f} | {c['pa_old']:.2f} | {c['pa_new']:.2f} |")
        lines.append("")
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data/rivers',
                    help='folder with one sub-folder per river holding orthophoto.tif, sampling_data.gpkg '
                         'and validation_sample.gpkg')
    ap.add_argument('--out', default='outputs/benchmark')
    ap.add_argument('--rivers', nargs='+', default=list(RIVERS))
    ap.add_argument('--block-size', type=int, default=50, help='block size in pixels (all rivers)')
    ap.add_argument('--block-m', type=float, default=None,
                    help='block size in metres; overrides --block-size, converted per river from the GSD')
    ap.add_argument('--coverage', type=float, default=0.9)
    ap.add_argument('--no-nested', action='store_true')
    ap.add_argument('--reuse-features', action='store_true')
    ap.add_argument('--eval-block-size', type=int, default=None,
                    help='px size of the validation squares; model blocks are aggregated to it (area-weighted majority)')
    ap.add_argument('--eval-only', action='store_true',
                    help='only re-run the design-based evaluation with the trained models already in --out')
    ap.add_argument('--smoothing', type=int, default=1, help='probability smoothing window (blocks); 1 = off')
    ap.add_argument('--fill', type=float, default=None, help='fill blocks with at least this valid fraction')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    z = norm.ppf(0.975)
    results_path = os.path.join(args.out, 'benchmark_results.json')
    # Results of rivers not run this time are kept, so the benchmark can be resumed river by river
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = {r['river']: r for r in json.load(f)}
    for name in args.rivers:
        print(f"\n{'#' * 70}\n# {name}\n{'#' * 70}")
        block_size = args.block_size
        if args.block_m is not None:
            with rasterio.open(os.path.join(args.data_root, name, 'orthophoto.tif')) as src:
                block_size = max(1, int(round(args.block_m / src.res[0])))
            print(f"Block size {args.block_m} m -> {block_size} px")
        if args.eval_only:
            results[name]['design_based'] = evaluate_river(name, args.data_root, os.path.join(args.out, name),
                                                           results[name]['block_size'], z, args.eval_block_size,
                                                           args.smoothing, args.fill)
        else:
            results[name] = run_river(name, args.data_root, args.out, block_size, args.coverage,
                                      not args.no_nested, z, args.reuse_features, args.eval_block_size,
                                      args.smoothing, args.fill)
        with open(results_path, 'w') as f:
            json.dump([results[r] for r in RIVERS if r in results], f, indent=2, default=float)
    write_summary([results[r] for r in RIVERS if r in results], os.path.join(args.out, 'benchmark_summary.md'))


if __name__ == '__main__':
    main()
