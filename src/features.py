import os
import pandas as pd
import geopandas as gpd
from fuzzywuzzy import fuzz
from .feature_extractor_masterv2 import extract_block_features
from .ml_utils import plot_class_distribution

def check_for_typos(gpkg_path, layer_name, threshold=85):
    """Checks for typos in the substrate labels."""
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
        if gdf.empty: return
        unique_classes = sorted(gdf['substrate'].dropna().unique())
        for i in range(len(unique_classes)):
            for j in range(i + 1, len(unique_classes)):
                if fuzz.ratio(unique_classes[i].lower(), unique_classes[j].lower()) >= threshold:
                    print(f"⚠️ Warning: Potential typo '{unique_classes[i]}' <--> '{unique_classes[j]}'")
    except Exception as e:
        print(f"Typo check failed: {e}")

def run_extraction(ortho_path, labels_path, layer_name, output_csv, block_size=50, roi_path=None, roi_layer_name='roi'):
    """Wraps the feature extraction process."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    check_for_typos(labels_path, layer_name)
    
    print(f"Extracting features from {ortho_path}...")
    data_df = extract_block_features(
        tile_size=block_size,
        labels_path=labels_path,
        layer_name=layer_name,
        orthophoto_path=ortho_path,
        roi_path=roi_path,
        roi_layer_name=roi_layer_name
    )
    
    if not data_df.empty:
        data_df.to_csv(output_csv, index=False)
        print(f"Extracted {len(data_df)} samples to {output_csv}")
        
        # Plot class distribution
        plot_class_distribution(data_df['class'])
        
        return data_df
    else:
        print("Extraction returned no data.")
        return None
