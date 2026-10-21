"""
Compare model configurations on the study rivers with the design-based benchmark.

Every configuration is trained (final model only, no CV) with several seeds; each trained model
is then scored on the existing validation samples for every post-processing variant
(probability smoothing window x edge filling). Results: one CSV row per
(river, configuration, seed, smoothing, fill) plus a summary averaged over seeds.

Usage (from the repository root):
    python -m benchmarks.run_experiments --experiment weighting --out outputs/phase2
"""
import argparse
import itertools
import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from benchmarks.run_benchmark import RIVERS, evaluate_on_old_sample, labels_at_points, load_validation, point_windows
from src.block_features import FEATURE_FAMILIES
from src.features import run_extraction
from src.modeling import train_pipeline

BASE = dict(class_weighting='balanced', weight_cap=10.0, per_polygon_weighting=False,
            selection_rule='upper', pfi_scoring='accuracy')        # Phase 1 behaviour

EXPERIMENTS = {
    'weighting': {
        'P1 balanced': {},
        'capped 10': dict(class_weighting='capped', weight_cap=10.0),
        'capped 5': dict(class_weighting='capped', weight_cap=5.0),
        'sqrt': dict(class_weighting='sqrt'),
        'capped 10 + polygon': dict(class_weighting='capped', weight_cap=10.0, per_polygon_weighting=True),
    },
    'selection': {
        'capped 10, upper/acc': dict(class_weighting='capped'),
        'capped 10, mean/bacc': dict(class_weighting='capped', selection_rule='mean', pfi_scoring='balanced_accuracy'),
        'capped 10, lower/bacc': dict(class_weighting='capped', selection_rule='lower', pfi_scoring='balanced_accuracy'),
    },
    'texture': {
        'P1 texture': dict(feature_families=['spectral', 'glcm', 'lbp']),
        'multi-scale texture': dict(feature_families=['spectral', 'glcm_ms', 'lbp_ms', 'gradient']),
        'all texture': dict(feature_families=list(FEATURE_FAMILIES)),
    },
}
# Experiments needing more than the default feature families get their own feature table
EXTRACT_FAMILIES = {'texture': tuple(FEATURE_FAMILIES)}
POSTPROCESS = [(1, None), (3, None), (5, None), (1, 0.5), (3, 0.5)]


def design_macro_f1(per_class):
    f1 = []
    for c in per_class:
        ua, pa = c['ua_new'], c['pa_new']
        if np.isnan(pa):        # class absent from the reference sample
            continue
        ua = 0.0 if np.isnan(ua) else ua
        f1.append(0.0 if ua + pa == 0 else 2 * ua * pa / (ua + pa))
    return float(np.mean(f1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', choices=list(EXPERIMENTS), required=True)
    ap.add_argument('--data-root', default='data/rivers',
                    help='folder with one sub-folder per river holding orthophoto.tif, sampling_data.gpkg '
                         'and validation_sample.gpkg')
    ap.add_argument('--out', default='outputs/phase2')
    ap.add_argument('--rivers', nargs='+', default=list(RIVERS))
    ap.add_argument('--block-size', type=int, default=50)
    ap.add_argument('--coverage', type=float, default=0.9)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 123])
    ap.add_argument('--base', default=None,
                    help='JSON overrides of the base configuration, e.g. \'{"class_weighting": "capped"}\'')
    args = ap.parse_args()
    base = {**BASE, **(json.loads(args.base) if args.base else {})}
    families = EXTRACT_FAMILIES.get(args.experiment)

    z = norm.ppf(0.975)
    rows_path = os.path.join(args.out, f'{args.experiment}_rows.csv')
    rows = []
    for river in args.rivers:
        river_out = os.path.join(args.out, river)
        csv = os.path.join(river_out, 'block_features_all.csv' if families else 'block_features.csv')
        if not os.path.exists(csv):
            run_extraction(os.path.join(args.data_root, river, 'orthophoto.tif'),
                           os.path.join(args.data_root, river, 'sampling_data.gpkg'), RIVERS[river]['layer'], csv,
                           block_size=args.block_size, min_label_coverage=args.coverage,
                           show_plot=False, **({'families': families} if families else {}))
        sample, old_names = load_validation(river, args.data_root)

        for (config, overrides), seed in itertools.product(EXPERIMENTS[args.experiment].items(), args.seeds):
            print(f"\n### {river} | {config} | seed {seed}", flush=True)
            model_dir = os.path.join(river_out, 'experiment_model')
            _, feats, _ = train_pipeline(csv, model_dir, evaluate=False, show_plots=False, random_state=seed,
                                         **{**base, **overrides})
            new_names = joblib.load(os.path.join(model_dir, 'final_class_mapping.joblib'))
            windows = point_windows(os.path.join(args.data_root, river, 'orthophoto.tif'), model_dir,
                                    args.block_size, sample, halo=2)
            for smoothing, fill in POSTPROCESS:
                pred = labels_at_points(windows, smoothing, fill)
                d = evaluate_on_old_sample(sample, pred, new_names, old_names, z)
                rows.append({
                    'river': river, 'config': config, 'seed': seed, 'smoothing': smoothing,
                    'fill': fill if fill is not None else 'none', 'n_features': len(feats),
                    'oa': d['oa_new']['estimate'], 'oa_se': d['oa_new']['se'],
                    'diff_vs_old_map': d['oa_diff']['estimate'],
                    'macro_f1': design_macro_f1(d['per_class']),
                    'n_nodata': d['n_nodata_new'],
                    'per_class': json.dumps({c['class']: [round(c['ua_new'], 3), round(c['pa_new'], 3)]
                                             for c in d['per_class']}),
                })
            pd.DataFrame(rows).to_csv(rows_path, index=False)

    df = pd.DataFrame(rows)
    summary = (df.groupby(['config', 'smoothing', 'fill', 'river'], sort=False)[['oa', 'macro_f1', 'n_features', 'n_nodata']]
               .mean().unstack('river'))
    avg = df.groupby(['config', 'smoothing', 'fill'], sort=False)[['oa', 'macro_f1']].mean()
    summary[('oa', 'MEAN')] = avg['oa']
    summary[('macro_f1', 'MEAN')] = avg['macro_f1']
    summary = summary.sort_index(axis=1)
    summary.to_csv(os.path.join(args.out, f'{args.experiment}_summary.csv'))
    pd.set_option('display.width', 250)
    print(summary.round(3).to_string())


if __name__ == '__main__':
    main()
