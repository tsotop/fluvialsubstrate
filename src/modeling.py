import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from .ml_utils import (
    reduce_features_by_correlation,
    select_features_by_importance_cv,
    report_model_performance_cv,
    get_model
)

def train_pipeline(features_csv, output_dir, model_type='RandomForest', 
                  use_spectral=True, use_glcm=True, use_lbp=True, 
                  use_class_weighting=True, corr_threshold=0.8, 
                  final_feature_count=None, n_splits=3, figsize=(12, 10)):
    """Train and evaluate the substrate classification pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(features_csv)
    df = df.dropna(subset=df.columns.difference(['label_id', 'block_row', 'block_col']))
    
    y_str = df['class']
    groups = df['label_id']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_str)

    # Adaptive fold count: min unique polygons in any class, clamped to [3, 10]
    min_cv_folds = 3
    max_cv_folds = 10
    polygons_per_class = df.groupby('class')['label_id'].nunique().sort_values()
    min_polygons_in_class = int(polygons_per_class.min())
    adaptive_splits = int(np.clip(min_polygons_in_class, min_cv_folds, max_cv_folds))

    print("\n--- Adaptive CV Fold Selection ---")
    print(f"Min unique polygons in a class: {min_polygons_in_class}")
    print(f"Using n_splits={adaptive_splits} (clamped to [{min_cv_folds}, {max_cv_folds}])")

    if min_polygons_in_class < min_cv_folds:
        print(
            f"⚠️ Least-supported class has only {min_polygons_in_class} polygons, "
            f"but minimum folds is forced to {min_cv_folds}."
        )
        print("   If CV splitter fails, collect more polygons for rare classes.")
    
    # Feature filtering
    spectral_cols = [c for c in df.columns if any(c.startswith(pre) for pre in ['mean_', 'std_', 'var_', 'norm_', 'c1_', 'c2_', 'c3_'])]
    glcm_cols = [c for c in df.columns if c.startswith('glcm_')]
    lbp_cols = [c for c in df.columns if c.startswith('lbp_')]
    
    selected_cols = []
    if use_spectral: selected_cols.extend(spectral_cols)
    if use_glcm: selected_cols.extend(glcm_cols)
    if use_lbp: selected_cols.extend(lbp_cols)
    
    X = df[selected_cols]

    # Correlation pruning before PFI (Spearman)
    corr_kept_features = reduce_features_by_correlation(X, corr_threshold)
    X_pruned = X[corr_kept_features]
    print(
        f"Correlation pruning: {len(X.columns)} -> {len(X_pruned.columns)} "
        f"features (threshold={corr_threshold})."
    )
    
    # Feature selection
    cv = StratifiedGroupKFold(n_splits=adaptive_splits, shuffle=True, random_state=42)
    importance_csv = os.path.join(output_dir, 'feature_importance.csv')
    
    features_to_keep, cv_scores = select_features_by_importance_cv(
        X_pruned, y_encoded, groups, cv, model_type, final_feature_count, 
        use_class_weighting, save_csv_path=importance_csv, figsize=figsize
    )
    
    # Run detailed performance report
    report_model_performance_cv(
        X_pruned[features_to_keep], y_encoded, groups, cv, 
        model_type, use_class_weighting, list(le.classes_)
    )
    
    X_final = X_pruned[features_to_keep].fillna(0)
    
    # Train final model
    num_classes = len(le.classes_)
    model = get_model(model_type, use_class_weighting, num_classes=num_classes)
    
    fit_params = {}
    if use_class_weighting:
        from sklearn.utils.class_weight import compute_sample_weight
        fit_params['sample_weight'] = compute_sample_weight('balanced', y=y_encoded)
        
    model.fit(X_final, y_encoded, **fit_params)
    
    # Save artifacts
    model_path = os.path.join(output_dir, 'final_model.joblib')
    feat_path = os.path.join(output_dir, 'final_model_features.joblib')
    map_path = os.path.join(output_dir, 'final_class_mapping.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(features_to_keep, feat_path)
    joblib.dump({i: cls for i, cls in enumerate(le.classes_)}, map_path)
    
    print(f"Model and artifacts saved to {output_dir}")
    print(f"--- Cross-Validation Results ---")
    print(f"Average Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    return model, features_to_keep, cv_scores
