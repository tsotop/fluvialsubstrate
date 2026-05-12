import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_importance(importance_csv, output_png, top_n=15, figsize=(10, 8)):
    """Logic from 06_plot_feat_importance.py (single site)."""
    df = pd.read_csv(importance_csv).sort_values(by='mean', ascending=False).head(top_n)
    
    def get_family(name):
        if name.startswith('glcm_'): return 'GLCM'
        if name.startswith('lbp_'): return 'LBP'
        return 'Spectral'

    colors = {'Spectral': '#4e79a7', 'GLCM': '#f28e2b', 'LBP': '#59a14f'}
    df['Family'] = df['feature'].apply(get_family)
    df['Color'] = df['Family'].map(colors)

    plt.figure(figsize=(10, 8))
    plt.barh(df['feature'], df['mean'], color=df['Color'], edgecolor='black', alpha=0.8)
    plt.gca().invert_yaxis()
    plt.title("Feature Importance", fontsize=14, fontweight='bold')
    plt.xlabel("Mean Decrease in Accuracy")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    handles = [plt.Rectangle((0,0),1,1, color=colors[f]) for f in colors]
    plt.legend(handles, colors.keys(), title="Feature Family")
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.show()
