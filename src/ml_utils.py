import yaml
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance
import time
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_config(config_path=None):
    """
    Loads configuration from a YAML file.
    Automatically finds the project root.
    """
    # Find project root (where 'configs' folder lives)
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
        
    # Inject project root into config for easy reference if needed
    config['project_root'] = project_root
    
    # Optional: Resolve all paths in 'paths' section relative to project root
    if 'paths' in config:
        for key, val in config['paths'].items():
            if isinstance(val, str) and not os.path.isabs(val):
                config['paths'][key] = os.path.join(project_root, val)
                
    return config

"""
Machine Learning Utilities
Contains helper functions for:
- Model initialization (RandomForest, XGBoost, LightGBM)
- Feature reduction (Correlation analysis)
- Feature selection (Permutation Feature Importance)
- Visualization (Class distribution, confusion matrices)
"""


def plot_class_distribution(y, figsize=(8, 6)):
    """
    Calculates and plots the distribution of classes.
    Expects labels (string or encoded).
    """
    print("\n--- Running Class Distribution Analysis ---")
    class_counts = y.value_counts().sort_values(ascending=False)
    
    plt.figure(figsize=figsize)
    plt.barh(class_counts.index, class_counts.values, color='skyblue', edgecolor='black', alpha=0.8)
    plt.gca().invert_yaxis() # Display largest class at the top
    plt.title('Class Distribution (Number of Blocks per Class)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Blocks (Support)')
    plt.ylabel('Substrate Class')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    print("Class counts details:")
    print(class_counts)
    return

def get_model(model_type, use_class_weighting, num_classes=None):
    """
    Initializes and returns a model instance based on type.
    'num_classes' is required for multiclass XGBoost/LightGBM.
    """
    # Base parameters common to most models
    model_params = {
        'random_state': 42,
        'n_jobs': -1
    }
    
    if model_type == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        model_params['n_estimators'] = 100
        if use_class_weighting:
            model_params['class_weight'] = 'balanced'
        return RandomForestClassifier(**model_params)
    
    elif model_type == 'XGBoost':
        from xgboost import XGBClassifier
        model_params['n_estimators'] = 300
        model_params['objective'] = 'multi:softmax'
        
        if num_classes is None:
            raise ValueError("XGBoost requires 'num_classes' for multiclass objective.")
        model_params['num_class'] = num_classes
        
        if use_class_weighting:
             print("Info: For XGBoost, class weighting will be handled via 'sample_weight' during fit.")
        return XGBClassifier(**model_params)

    elif model_type == 'LightGBM':
        from lightgbm import LGBMClassifier
        model_params['n_estimators'] = 300
        if use_class_weighting:
            model_params['class_weight'] = 'balanced'
        
        if num_classes is not None:
            model_params['objective'] = 'multiclass'
            model_params['num_class'] = num_classes
        return LGBMClassifier(**model_params)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def reduce_features_by_correlation(df_features, corr_threshold):
    """
    Reduces the number of features by removing highly correlated ones.
    """
    print("\n--- Running Correlation Analysis ---")
    # Drop constant columns first to avoid errors in correlation calculation
    non_constant_cols = df_features.columns[df_features.nunique() > 1]
    df_features_filt = df_features[non_constant_cols]
    if len(non_constant_cols) < len(df_features.columns):
          print(f"Dropped {len(df_features.columns) - len(non_constant_cols)} constant columns before correlation.")
        
    corr_matrix = df_features_filt.corr(method='spearman').abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]
    print(f"Identified {len(to_drop)} columns to drop (Spearman > {corr_threshold}): {to_drop}")
    
    features_to_keep = df_features_filt.drop(columns=to_drop).columns.tolist()
    print(f"Features remaining after correlation filter: {len(features_to_keep)}")
    
    return features_to_keep

def select_features_by_importance_cv(X, y, groups, cv_splitter, model_type, n_features_to_keep, use_class_weighting, save_csv_path=None, figsize=(12, 10)):
    """
    Selects the top k features using robust, cross-validated
    Permutation Feature Importance (PFI).
    
    Args:
        X (pd.DataFrame): Feature data.
        y (np.ndarray): Encoded target labels (integers).
        groups (pd.Series): Group labels for CV.
        cv_splitter: Cross-validation object (e.g., StratifiedGroupKFold).
        model_type (str): Type of model ('RandomForest', 'XGBoost', 'LightGBM').
        n_features_to_keep (int, optional): Number of top features to select. 
            If None, all features with upper bound (mean + std) > 0 are kept.
        use_class_weighting (bool): Whether to enable class weighting.
        save_csv_path (str, optional): Path to save the feature importance CSV.
        
    Returns:
        tuple: (list of top k features, list of CV scores)
    """
    print(f"\n--- Running Cross-Validated Permutation Importance for {model_type} ---")
    
    all_importances = []
    cv_scores = []
    
    # Check if y is numpy array
    if not isinstance(y, np.ndarray):
         raise TypeError(f"Expected y to be a NumPy array (encoded labels), but got {type(y)}")
         
    for i, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y, groups)):
        print(f"--- PFI Fold {i+1}/{cv_splitter.get_n_splits()} ---")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        
        # Use standard NumPy indexing for the NumPy array y
        y_train, y_test = y[train_idx], y[test_idx]
        
        num_classes_fold = len(np.unique(y_train))
        model_fold = get_model(model_type, use_class_weighting, num_classes=num_classes_fold)
        
        # Handle sample weights for imbalanced models
        fit_params = {}
        if use_class_weighting:
            from sklearn.utils.class_weight import compute_sample_weight
            try:
                fit_params['sample_weight'] = compute_sample_weight(
                    class_weight='balanced',
                    y=y_train
                )
            except ValueError:
                    pass
        
        # Fill NaNs before fitting
        X_train_filled = X_train.fillna(0)
        X_test_filled = X_test.fillna(0)

        model_fold.fit(X_train_filled, y_train, **fit_params)
        
        # Calculate permutation importance on the test set
        start_time = time.time()
        pfi_result = permutation_importance(
            model_fold, X_test_filled, y_test, 
            n_repeats=5, random_state=42, n_jobs=-1, scoring='accuracy'
        )
        elapsed = time.time() - start_time
        print(f"    PFI calculation complete in {elapsed:.2f}s")
        
        # Save the score for reporting
        fold_score = model_fold.score(X_test_filled, y_test)
        cv_scores.append(fold_score)
        print(f"    Fold Accuracy: {fold_score:.4f}")
        
        all_importances.append(pfi_result.importances_mean)

    # --- Aggregate and rank features ---
    mean_importances = np.mean(all_importances, axis=0)
    std_importances = np.std(all_importances, axis=0)
    
    importances_df = pd.DataFrame({
        'feature': X.columns,
        'mean': mean_importances,
        'std': std_importances
    }).sort_values(by='mean', ascending=False)
    
    # *** NEW: Save CSV if path provided ***
    if save_csv_path:
        importances_df.to_csv(save_csv_path, index=False)
        print(f"✅ Saved feature importance table to: {save_csv_path}")
    
    # --- Plotting ---
    print("\nGenerating feature importance plot...")
    plot_df = importances_df.head(25) # Plot top 25 for readability
    
    plt.figure(figsize=figsize)
    plt.barh(
        plot_df['feature'], 
        plot_df['mean'], 
        xerr=plot_df['std'], 
        capsize=4, 
        color='skyblue',
        edgecolor='black'
    )
    plt.gca().invert_yaxis() # Display top feature at the top
    plt.title(f'Mean Permutation Importance ({model_type}) across {cv_splitter.get_n_splits()} Folds (Top {len(plot_df)})')
    plt.xlabel('Mean Importance (Drop in Accuracy)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()
    

    # --- Select features based on statistical requirement ---
    # Requirement: mean + std > 0 (upper bound must be positive)
    importances_df['upper_bound'] = importances_df['mean'] + importances_df['std']
    
    threshold_passing = importances_df[importances_df['upper_bound'] > 0]
    total_potential = len(importances_df)
    passing_count = len(threshold_passing)
    
    print(f"\nFeature selection results (Requirement: Mean + Std > 0):")
    print(f"- {passing_count} features out of {total_potential} passed the statistical test.")
    
    if n_features_to_keep is not None:
        print(f"- Limiting to top {n_features_to_keep} features as requested.")
        final_features = threshold_passing.head(n_features_to_keep)['feature'].tolist()
    else:
        final_features = threshold_passing['feature'].tolist()
    
    print(f"\nSelected {len(final_features)} features:")
    print(final_features)
    
    return final_features, cv_scores

def report_model_performance_cv(X, y, groups, cv_splitter, model_type, use_class_weighting, class_names):
    """
    Calculates and prints detailed cross-validation metrics (Precision, Recall, F1).
    """
    print(f"\n--- Running Detailed Cross-Validated Performance Report for {model_type} ---")
    
    fold_reports = []
    
    for i, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx].fillna(0), X.iloc[test_idx].fillna(0)
        y_train, y_test = y[train_idx], y[test_idx]
        
        num_classes_fold = len(np.unique(y_train))
        model_fold = get_model(model_type, use_class_weighting, num_classes=num_classes_fold)
        
        fit_params = {}
        if use_class_weighting:
            from sklearn.utils.class_weight import compute_sample_weight
            try:
                fit_params['sample_weight'] = compute_sample_weight(class_weight='balanced', y=y_train)
            except ValueError:
                pass

        model_fold.fit(X_train, y_train, **fit_params)
        y_pred = model_fold.predict(X_test)
        
        # Get dictionary report
        # Fix: explicitly provide labels to handle folds with missing classes
        report = classification_report(
            y_test, y_pred, 
            labels=np.arange(len(class_names)),
            target_names=class_names, 
            output_dict=True, 
            zero_division=0
        )
        fold_reports.append(report)
        print(f"    Fold {i+1}/{cv_splitter.get_n_splits()} complete.")

    # Aggregate metrics across folds
    agg_report = {}
    for cls in list(class_names) + ['macro avg', 'weighted avg']:
        if cls in fold_reports[0]:
            agg_report[cls] = {
                'precision': np.mean([r[cls]['precision'] for r in fold_reports]),
                'recall': np.mean([r[cls]['recall'] for r in fold_reports]),
                'f1-score': np.mean([r[cls]['f1-score'] for r in fold_reports]),
                'support': np.sum([r[cls].get('support', 0) for r in fold_reports])
            }
    
    # Print a nice table
    print("\n" + "="*65)
    print(f"{'Class':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 65)
    for cls in list(agg_report.keys()):
        m = agg_report[cls]
        if cls in ['macro avg', 'weighted avg']: print("-" * 65)
        print(f"{cls:<25} | {m['precision']:<10.3f} | {m['recall']:<10.3f} | {m['f1-score']:<10.3f}")
    print("="*65)
    
    return agg_report