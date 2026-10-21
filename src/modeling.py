import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from .block_features import FEATURE_FAMILIES
from .features import feature_meta_path
from .ml_utils import (
    adaptive_n_splits,
    select_features,
    plot_feature_importance,
    cross_val_predict_grouped,
    report_pooled,
    summarize,
    get_model,
    fit_model,
    Weighting,
)


def polygon_agreement(groups, y, y_pred, class_names):
    """Per polygon: share of its blocks predicted as its own class in CV, and the most predicted class."""
    rows = []
    df = pd.DataFrame({'g': groups, 'y': y, 'p': y_pred})
    for g, d in df.groupby('g'):
        top = d['p'].value_counts()
        rows.append({'label_id': int(g), 'class': class_names[d['y'].iloc[0]], 'n_blocks': len(d),
                     'agreement': float((d['p'] == d['y']).mean()),
                     'most_predicted': class_names[top.index[0]], 'most_predicted_share': float(top.iloc[0] / len(d))})
    return rows


def train_pipeline(features_csv, output_dir, model_type='RandomForest',
                   use_spectral=True, use_glcm=True, use_lbp=True, feature_families=None,
                   class_weighting='capped', weight_cap=10.0, per_polygon_weighting=False,
                   corr_threshold=0.9, final_feature_count=None,
                   selection_rule='mean', pfi_scoring='balanced_accuracy',
                   nested_cv=True, random_cv=True, evaluate=True,
                   random_state=42, show_plots=True, figsize=(12, 10), use_class_weighting=None):
    """
    Train and evaluate the substrate classification pipeline.

    Evaluation (spatial CV, grouped by polygon, pooled out-of-fold predictions):
      - nested_cv=True: feature selection is repeated inside every training fold, so the
        reported scores are not inflated by selecting features on the test polygons.
      - nested_cv=False: features are selected once on all data, then cross-validated
        (faster, optimistic).
      - random_cv=True: also reports a non-spatial StratifiedKFold on the same features;
        the difference to spatial CV is the "optimism gap".
    The final model is trained on all data with the features selected on all data.

    Class weighting (see ml_utils.Weighting): 'none', 'balanced', 'capped' (rare classes weigh at
    most `weight_cap` times the largest class) or 'sqrt'; `per_polygon_weighting` gives every
    polygon the same total weight within its class. `use_class_weighting` (bool) is the legacy
    switch: True -> 'balanced', False -> 'none'.

    Feature selection: PFI scored with `pfi_scoring`, kept by `selection_rule` ('lower', 'mean'
    or 'upper'; see ml_utils.select_features). evaluate=False skips all cross-validation and only
    fits the final model (for quick experiments).
    """
    if use_class_weighting is not None:
        class_weighting = 'balanced' if use_class_weighting else 'none'
    weighting = Weighting(class_weighting, weight_cap, per_polygon_weighting)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(features_csv)
    if feature_families is None:
        # All families present in the feature table; the legacy switches drop family groups
        off = {fam for flag, fams in ((use_spectral, ['spectral']), (use_glcm, ['glcm', 'glcm_ms']),
                                      (use_lbp, ['lbp', 'lbp_ms'])) if not flag for fam in fams}
        feature_families = [fam for fam in FEATURE_FAMILIES if fam not in off]
    selected_cols = [c for fam in feature_families for c in FEATURE_FAMILIES[fam] if c in df.columns]
    df = df.dropna(subset=selected_cols + ['class'])

    le = LabelEncoder()
    y = le.fit_transform(df['class'])
    class_names = list(le.classes_)
    num_classes = len(class_names)
    groups = df['label_id'].to_numpy()
    X = df[selected_cols].reset_index(drop=True)

    n_splits, min_polygons = adaptive_n_splits(y, groups)
    print("\n--- Adaptive CV Fold Selection ---")
    print(f"Min unique polygons in a class: {min_polygons}")
    print(f"Using n_splits={n_splits} (clamped to [3, 10])")
    if min_polygons < 3:
        print(f"⚠️ Least-supported class has only {min_polygons} polygons; some folds will not test it. "
              "Collect more polygons for rare classes.")

    def selector(X_tr, y_tr, g_tr, verbose=False):
        feats, _, _ = select_features(
            X_tr, y_tr, g_tr, model_type, weighting, num_classes, corr_threshold,
            final_feature_count, random_state=random_state, selection_rule=selection_rule,
            pfi_scoring=pfi_scoring, verbose=verbose
        )
        return feats

    # --- Feature selection on all data (defines the final model) ---
    features_to_keep, importances_df, _ = select_features(
        X, y, groups, model_type, weighting, num_classes, corr_threshold,
        final_feature_count, n_splits=n_splits, random_state=random_state, selection_rule=selection_rule,
        pfi_scoring=pfi_scoring, verbose=True
    )
    importance_csv = os.path.join(output_dir, 'feature_importance.csv')
    importances_df.to_csv(importance_csv, index=False)
    print(f"✅ Saved feature importance table to: {importance_csv}")
    if show_plots:
        plot_feature_importance(importances_df, f'Permutation Importance ({model_type}), '
                                f'{importances_df["n_estimates"].iloc[0]} estimates per feature', figsize)

    results = {
        'classes': class_names,
        'n_samples': int(len(y)),
        'n_polygons': int(len(np.unique(groups))),
        'n_splits': n_splits,
        'selected_features': features_to_keep,
    }

    def run_cv():
        # --- Spatial CV ---
        spatial_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if nested_cv:
            print(f"\n--- Nested spatial CV ({n_splits} outer folds, feature selection inside each fold) ---")
            y_pred, fold_info = cross_val_predict_grouped(
                X, y, groups, spatial_cv, model_type, weighting, num_classes,
                feature_selector=selector, random_state=random_state
            )
            spatial_title = "Spatial CV (nested, pooled out-of-fold)"
        else:
            print(f"\n--- Spatial CV ({n_splits} folds, features selected on all data: optimistic) ---")
            y_pred, fold_info = cross_val_predict_grouped(
                X[features_to_keep], y, groups, spatial_cv, model_type, weighting, num_classes,
                random_state=random_state
            )
            spatial_title = "Spatial CV (post-selection, pooled out-of-fold)"
        spatial_report = report_pooled(y, y_pred, class_names, spatial_title)

        results['spatial_cv'] = {'nested': nested_cv, **summarize(spatial_report), 'folds': fold_info,
                                 'report': spatial_report,
                                 'confusion': confusion_matrix(y, y_pred, labels=np.arange(num_classes)).tolist(),
                                 'polygons': polygon_agreement(groups, y, y_pred, class_names)}

        # --- Non-spatial CV (optimism gap) ---
        if random_cv:
            print(f"\n--- Standard (non-spatial) CV ({n_splits} folds) ---")
            random_split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            y_pred_r, fold_info_r = cross_val_predict_grouped(
                X[features_to_keep], y, groups, random_split, model_type, weighting, num_classes,
                random_state=random_state
            )
            random_report = report_pooled(y, y_pred_r, class_names, "Standard CV (pooled out-of-fold)")
            results['standard_cv'] = {**summarize(random_report), 'folds': fold_info_r, 'report': random_report}
            gap = results['standard_cv']['macro_f1'] - results['spatial_cv']['macro_f1']
            print(f"\nOptimism gap (macro F1, standard - spatial): {gap:+.3f}")

    if evaluate:
        run_cv()

    # --- Final model ---
    X_final = X[features_to_keep]
    model = get_model(model_type, num_classes, random_state)
    fit_model(model, X_final, y, weighting, groups)

    feature_meta = {}
    if os.path.exists(feature_meta_path(features_csv)):
        with open(feature_meta_path(features_csv)) as f:
            feature_meta = json.load(f)

    joblib.dump(model, os.path.join(output_dir, 'final_model.joblib'))
    joblib.dump(features_to_keep, os.path.join(output_dir, 'final_model_features.joblib'))
    joblib.dump({i: cls for i, cls in enumerate(class_names)}, os.path.join(output_dir, 'final_class_mapping.joblib'))
    with open(os.path.join(output_dir, 'final_model_meta.json'), 'w') as f:
        json.dump({
            'model_type': model_type,
            'block_size': feature_meta.get('block_size'),
            'feature_version': feature_meta.get('feature_version'),
            'features': features_to_keep,
            'classes': class_names,
            'class_weighting': class_weighting,
            'weight_cap': weight_cap,
            'per_polygon_weighting': per_polygon_weighting,
            'selection_rule': selection_rule,
            'pfi_scoring': pfi_scoring,
            'corr_threshold': corr_threshold,
        }, f, indent=2)
    with open(os.path.join(output_dir, 'cv_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nModel and artifacts saved to {output_dir}")
    return model, features_to_keep, results
