import os
import time
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, accuracy_score

"""
Machine Learning Utilities
Contains helper functions for:
- Model initialization (RandomForest, XGBoost, LightGBM)
- Feature reduction (Correlation analysis)
- Feature selection (Permutation Feature Importance)
- Cross-validated performance (pooled out-of-fold predictions)
"""


def load_config(config_path=None):
    """
    Loads configuration from a YAML file.
    Automatically finds the project root.
    """
    import yaml
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))

    if config_path is None:
        config_path = os.path.join(project_root, "configs", "default.yaml")
    elif not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['project_root'] = project_root
    if 'paths' in config:
        for key, val in config['paths'].items():
            if isinstance(val, str) and not os.path.isabs(val):
                config['paths'][key] = os.path.join(project_root, val)

    return config


def plot_class_distribution(y, figsize=(8, 6)):
    """
    Calculates and plots the distribution of classes.
    Expects labels (string or encoded).
    """
    print("\n--- Running Class Distribution Analysis ---")
    class_counts = y.value_counts().sort_values(ascending=False)

    plt.figure(figsize=figsize)
    plt.barh(class_counts.index, class_counts.values, color='skyblue', edgecolor='black', alpha=0.8)
    plt.gca().invert_yaxis()
    plt.title('Class Distribution (Number of Blocks per Class)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Blocks (Support)')
    plt.ylabel('Substrate Class')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("Class counts details:")
    print(class_counts)


def get_model(model_type, num_classes=None, random_state=42):
    """
    Initializes and returns a model instance based on type.
    Class imbalance is handled through `fit_model(..., weighting)`, never through
    the estimator's own class_weight (the two would multiply).
    """
    model_params = {'random_state': random_state, 'n_jobs': -1}

    if model_type == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, **model_params)

    elif model_type == 'XGBoost':
        from xgboost import XGBClassifier
        if num_classes is None:
            raise ValueError("XGBoost requires 'num_classes' for multiclass objective.")
        return XGBClassifier(n_estimators=300, objective='multi:softmax', num_class=num_classes, **model_params)

    elif model_type == 'LightGBM':
        from lightgbm import LGBMClassifier
        if num_classes is not None:
            model_params.update(objective='multiclass', num_class=num_classes)
        return LGBMClassifier(n_estimators=300, verbose=-1, **model_params)

    raise ValueError(f"Unknown model_type: {model_type}")


@dataclass(frozen=True)
class Weighting:
    """
    Per-sample training weights.

    scheme:
      'none'     : all blocks weigh 1.
      'balanced' : class weight = N_max / N_c (every class has the same total weight).
      'capped'   : like 'balanced', but no class weighs more than `cap` times the largest class.
      'sqrt'     : class weight = sqrt(N_max / N_c).
    per_polygon : within a class, every polygon gets the same total weight, so one large
                  polygon cannot dominate its class.
    """
    scheme: str = 'capped'
    cap: float = 10.0
    per_polygon: bool = False

    def sample_weights(self, y, groups=None):
        y = np.asarray(y)
        w = np.ones(len(y), dtype=float)
        if self.per_polygon and groups is not None:
            df = pd.DataFrame({'y': y, 'g': np.asarray(groups)})
            blocks_in_poly = df.groupby(['y', 'g'])['y'].transform('size')
            polys_in_class = df.groupby('y')['g'].transform('nunique')
            blocks_in_class = df.groupby('y')['y'].transform('size')
            w = (blocks_in_class / (polys_in_class * blocks_in_poly)).to_numpy(dtype=float)
        if self.scheme == 'none':
            return w
        classes, counts = np.unique(y, return_counts=True)
        ratio = counts.max() / counts
        if self.scheme == 'balanced':
            factor = ratio
        elif self.scheme == 'capped':
            factor = np.minimum(ratio, self.cap)
        elif self.scheme == 'sqrt':
            factor = np.sqrt(ratio)
        else:
            raise ValueError(f"Unknown weighting scheme: {self.scheme}")
        w = w * pd.Series(factor, index=classes).loc[y].to_numpy()
        return w * len(w) / w.sum()

    @classmethod
    def parse(cls, value):
        """Accepts a Weighting, a bool (legacy use_class_weighting) or a scheme name."""
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls('balanced' if value else 'none')
        return cls(str(value))


def fit_model(model, X, y, weighting, groups=None):
    """Fits a model with per-sample weights from `weighting` (a Weighting)."""
    return model.fit(X, y, sample_weight=weighting.sample_weights(y, groups))


def adaptive_n_splits(y, groups, lo=3, hi=10):
    """Fold count = polygons in the least represented class, clamped to [lo, hi]."""
    per_class = pd.Series(groups).groupby(np.asarray(y)).nunique()
    return int(np.clip(per_class.min(), lo, hi)), int(per_class.min())


def reduce_features_by_correlation(df_features, corr_threshold, verbose=True):
    """
    Reduces the number of features by removing highly correlated ones.
    For each correlated pair the feature appearing first in column order is kept.
    """
    non_constant_cols = df_features.columns[df_features.nunique() > 1]
    df_features_filt = df_features[non_constant_cols]

    corr_matrix = df_features_filt.corr(method='spearman').abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]
    features_to_keep = df_features_filt.drop(columns=to_drop).columns.tolist()

    if verbose:
        print("\n--- Running Correlation Analysis ---")
        if len(non_constant_cols) < len(df_features.columns):
            print(f"Dropped {len(df_features.columns) - len(non_constant_cols)} constant columns before correlation.")
        print(f"Identified {len(to_drop)} columns to drop (Spearman > {corr_threshold}): {to_drop}")
        print(f"Features remaining after correlation filter: {len(features_to_keep)}")

    return features_to_keep


def permutation_importance_cv(X, y, groups, cv_splitter, model_type, weighting, num_classes, n_repeats=5,
                              random_state=42, scoring='balanced_accuracy', verbose=True):
    """
    Permutation Feature Importance on the held-out fold of every CV split.
    Every (fold, repeat) pair is one estimate, so each feature gets n_repeats x K
    estimates; mean and std are taken over all of them.
    """
    raw = []
    fold_scores = []
    for i, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y, groups)):
        model = get_model(model_type, num_classes, random_state)
        fit_model(model, X.iloc[train_idx], y[train_idx], weighting, np.asarray(groups)[train_idx])

        start_time = time.time()
        pfi = permutation_importance(
            model, X.iloc[test_idx], y[test_idx],
            n_repeats=n_repeats, random_state=random_state, n_jobs=-1, scoring=scoring
        )
        raw.append(pfi.importances)
        fold_scores.append(model.score(X.iloc[test_idx], y[test_idx]))
        if verbose:
            print(f"--- PFI Fold {i+1}/{cv_splitter.get_n_splits()}: accuracy {fold_scores[-1]:.4f} "
                  f"({time.time() - start_time:.1f}s)")

    raw = np.concatenate(raw, axis=1)
    importances_df = pd.DataFrame({
        'feature': X.columns,
        'mean': raw.mean(axis=1),
        'std': raw.std(axis=1),
        'n_estimates': raw.shape[1],
    }).sort_values(by='mean', ascending=False).reset_index(drop=True)
    return importances_df, fold_scores


SELECTION_RULES = {
    'upper': ('mean + std > 0', lambda df: df['mean'] + df['std'] > 0),
    'mean': ('mean > 0', lambda df: df['mean'] > 0),
    'lower': ('mean - std > 0', lambda df: df['mean'] - df['std'] > 0),
}


def select_features(X, y, groups, model_type, weighting, num_classes, corr_threshold, n_features_to_keep=None,
                    n_splits=None, random_state=42, selection_rule='mean', pfi_scoring='balanced_accuracy',
                    min_features=5, verbose=True):
    """
    Correlation pruning followed by cross-validated PFI selection.

    selection_rule: 'upper' keeps features whose mean + std importance is positive (permissive),
    'mean' those with positive mean, 'lower' those whose mean - std is positive (strict).
    At least `min_features` (the most important ones) are always kept.

    Returns (selected features, importance table, fold accuracies).
    """
    corr_kept = reduce_features_by_correlation(X, corr_threshold, verbose=verbose)
    if n_splits is None:
        n_splits, _ = adaptive_n_splits(y, groups)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if verbose:
        print(f"\n--- Running Cross-Validated Permutation Importance for {model_type} ({n_splits} folds, "
              f"scoring={pfi_scoring}) ---")
    importances_df, fold_scores = permutation_importance_cv(
        X[corr_kept], y, groups, cv, model_type, weighting, num_classes, random_state=random_state,
        scoring=pfi_scoring, verbose=verbose
    )

    rule_text, rule = SELECTION_RULES[selection_rule]
    passes = rule(importances_df)
    passes |= importances_df.index < min_features      # table is sorted by mean importance
    importances_df['selected'] = passes
    passing = importances_df[passes]
    if n_features_to_keep is not None:
        passing = passing.head(n_features_to_keep)
    selected = passing['feature'].tolist()

    if verbose:
        print(f"\nFeature selection results (Requirement: {rule_text}, at least {min_features}):")
        print(f"- {len(selected)} features out of {len(importances_df)} selected.")
        print(selected)
    return selected, importances_df, fold_scores


def plot_feature_importance(importances_df, title, figsize=(12, 10), top_n=25):
    plot_df = importances_df.head(top_n)
    plt.figure(figsize=figsize)
    plt.barh(plot_df['feature'], plot_df['mean'], xerr=plot_df['std'], capsize=4,
             color='skyblue', edgecolor='black')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel('Mean Importance (Drop in Score)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()


def cross_val_predict_grouped(X, y, groups, cv_splitter, model_type, weighting, num_classes,
                              feature_selector=None, random_state=42):
    """
    Out-of-fold predictions for every sample.

    If `feature_selector(X_train, y_train, groups_train) -> features` is given it is run
    inside every training fold (nested CV), so the held-out fold never influences
    which features are used to predict it.
    """
    y_pred = np.full(len(y), -1)
    fold_info = []
    for i, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        feats = list(X.columns)
        if feature_selector is not None:
            feats = feature_selector(X_train, y_train, np.asarray(groups)[train_idx])
        model = get_model(model_type, num_classes, random_state)
        fit_model(model, X_train[feats], y_train, weighting, np.asarray(groups)[train_idx])
        y_pred[test_idx] = model.predict(X.iloc[test_idx][feats])
        fold_info.append({
            'fold': i + 1,
            'n_test': len(test_idx),
            'accuracy': accuracy_score(y[test_idx], y_pred[test_idx]),
            'n_features': len(feats),
        })
        print(f"    Fold {i+1}/{cv_splitter.get_n_splits()}: accuracy {fold_info[-1]['accuracy']:.4f}, "
              f"{len(feats)} features")
    return y_pred, fold_info


def report_pooled(y_true, y_pred, class_names, title):
    """Prints and returns a classification report computed on pooled out-of-fold predictions."""
    report = classification_report(
        y_true, y_pred, labels=np.arange(len(class_names)), target_names=class_names,
        output_dict=True, zero_division=0
    )
    print("\n" + "=" * 65)
    print(title)
    print("-" * 65)
    print(f"{'Class':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 65)
    for cls in list(class_names) + ['macro avg', 'weighted avg']:
        m = report[cls]
        if cls == 'macro avg': print("-" * 65)
        print(f"{cls:<25} | {m['precision']:<10.3f} | {m['recall']:<10.3f} | {m['f1-score']:<10.3f}")
    print("-" * 65)
    print(f"{'Overall accuracy':<25} | {report['accuracy']:.3f}")
    print("=" * 65)
    return report


def summarize(report):
    return {
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
    }
