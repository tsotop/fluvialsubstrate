import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from skimage.color import rgb2lab, rgb2hsv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tqdm import tqdm
import importlib
from skimage.transform import downscale_local_mean
from .roi_utils import load_optional_roi

"""
Master Feature Extractor Library (v2)
Contains functions to calculate color invariants, GLCM textures, and LBP features
from orthophoto blocks within labeled polygons.
"""

# --- LBP Parameters ---
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = 'uniform'
LBP_N_BINS = LBP_POINTS + 2

def get_color_invariants(rgb_patch):
    """Calculates the mean c1, c2, c3 color invariant features for a patch."""
    epsilon = 1e-6
    R, G, B = rgb_patch[:, :, 0].astype(float), rgb_patch[:, :, 1].astype(float), rgb_patch[:, :, 2].astype(float)
    c1 = np.mean(np.arctan(R / (np.maximum(G, B) + epsilon)))
    c2 = np.mean(np.arctan(G / (np.maximum(R, B) + epsilon)))
    c3 = np.mean(np.arctan(B / (np.maximum(R, G) + epsilon)))
    return {'c1_invariant': c1, 'c2_invariant': c2, 'c3_invariant': c3}

def get_glcm_features(gray_patch, prefix):
    """Calculates all GLCM properties for a given grayscale patch."""
    gray_patch_int = gray_patch.astype(np.uint8)
    # Ensure levels is appropriate, handle small range of values
    max_val = np.max(gray_patch_int)
    levels = 256 if max_val > 0 else 1 # Avoid error on empty patch
    if max_val > 0 and max_val < 255:
        levels = max_val + 1
        
    glcm = graycomatrix(gray_patch_int, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-6))
    
    return {
        f'glcm_{prefix}_contrast': contrast, f'glcm_{prefix}_dissimilarity': dissimilarity,
        f'glcm_{prefix}_homogeneity': homogeneity, f'glcm_{prefix}_ASM': asm,
        f'glcm_{prefix}_energy': energy, f'glcm_{prefix}_correlation': correlation,
        f'glcm_{prefix}_entropy': entropy
    }

def get_lbp_histogram(gray_patch, prefix):
    """Calculates the LBP histogram for a given grayscale patch."""
    gray_patch_int = gray_patch.astype(np.uint8)
    lbp = local_binary_pattern(gray_patch_int, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
    (hist, _) = np.histogram(
        lbp.ravel(), density=True,
        bins=np.arange(0, LBP_N_BINS + 1), range=(0, LBP_N_BINS)
    )
    return {f'lbp_{prefix}_bin_{i}': val for i, val in enumerate(hist)}

def extract_block_features(tile_size, labels_path, layer_name, orthophoto_path,
                           include_spectral=True, include_glcm=True, include_lbp=True,
                           roi_path=None, roi_layer_name='roi'):
    """
    Extracts block features for labeled polygons.
    """
    all_features = []
    
    print(f"Reading labels from: {labels_path} (Layer: {layer_name})")
    try:
        labels_shp = gpd.read_file(labels_path, layer=layer_name)
    except Exception as e:
        print(f"--- ERROR ---")
        print(f"Could not read layer '{layer_name}' from file '{labels_path}'.")
        print(f"Error: {e}")
        print("Please ensure the file path is correct and the layer exists.")
        return pd.DataFrame() # Return empty dataframe

    if labels_shp.empty:
        print(f"Warning: No features found in layer '{layer_name}'. Returning empty DataFrame.")
        return pd.DataFrame()

    roi_gdf = load_optional_roi(roi_path or labels_path, roi_layer_name)
    roi_union = None if roi_gdf is None else roi_gdf.geometry.unary_union

    # These will be dynamically populated
    spectral_cols = []
    glcm_cols = []
    lbp_cols = []

    with rasterio.open(orthophoto_path) as src:
        if labels_shp.crs != src.crs:
            print(f"Reprojecting labels from {labels_shp.crs} to {src.crs}...")
            labels_shp = labels_shp.to_crs(src.crs)

        if roi_gdf is not None and roi_gdf.crs != src.crs:
            roi_gdf = roi_gdf.to_crs(src.crs)
            roi_union = roi_gdf.geometry.unary_union

        if roi_union is not None:
            labels_shp = labels_shp[labels_shp.geometry.intersects(roi_union)].copy()
            if labels_shp.empty:
                print("Warning: No label geometries intersect the ROI. Returning empty DataFrame.")
                return pd.DataFrame()
            
        desc = f"Extracting ({'S' if include_spectral else ''}{'G' if include_glcm else ''}{'L' if include_lbp else ''}) Features"
        
        # Use .itertuples() for efficient iteration, which includes the index (fid)
        for row in tqdm(labels_shp.itertuples(), total=len(labels_shp), desc=desc):
            geom = row.geometry
            label_fid = row.Index  # Get the GeoPackage 'fid' from the GeoDataFrame index
            label_class = row.substrate

            if roi_union is not None:
                geom = geom.intersection(roi_union)
                if geom.is_empty:
                    continue
            
            if geom is None or geom.is_empty: continue
            try:
                out_image_raw, out_transform = mask(src, [geom], crop=True, all_touched=True)
            except ValueError as e:
                print(f"Skipping geometry {label_fid} (likely out of bounds): {e}")
                continue
                
            valid_mask = (out_image_raw.sum(axis=0) != 0) 
            if not valid_mask.any(): continue
                
            rgb_patch_full = np.transpose(out_image_raw[0:3, :, :], (1, 2, 0))
            
            # Pre-calculate full-image channels only if needed
            if include_spectral or include_glcm or include_lbp:
                lab_patch_full = rgb2lab(rgb_patch_full)
                hsv_patch_full = rgb2hsv(rgb_patch_full)
                patch_L_star_full = lab_patch_full[:, :, 0]
                patch_HLS_L_full = (hsv_patch_full[:, :, 2] * 255).astype(np.uint8)

            block_mask = downscale_local_mean(valid_mask.astype(float), (tile_size, tile_size)) > 0
            
            new_H, new_W = block_mask.shape
            H_full, W_full, _ = rgb_patch_full.shape

            for r in range(new_H):
                for c in range(new_W):
                    if block_mask[r, c] == 0: continue
                        
                    r_start, c_start = r * tile_size, c * tile_size
                    r_end, c_end = min(r_start + tile_size, H_full), min(c_start + tile_size, W_full)
                    
                    rgb_patch = rgb_patch_full[r_start:r_end, c_start:c_end]
                    
                    if not rgb_patch.any(): continue

                    # Base features
                    features = {
                        'label_id': label_fid, 'block_row': r, 'block_col': c,
                        'class': label_class
                    }
                    
                    # Pre-calculate block-level channels if needed
                    if include_spectral or include_glcm or include_lbp:
                        patch_HLS_L = patch_HLS_L_full[r_start:r_end, c_start:c_end]
                        patch_L_star = patch_L_star_full[r_start:r_end, c_start:c_end]
                        lab_patch_block = lab_patch_full[r_start:r_end, c_start:c_end]

                    if include_spectral:
                        mean_rgb = np.mean(rgb_patch, axis=(0, 1))
                        mean_lab_l = np.mean(patch_L_star)
                        mean_hls_l = np.mean(patch_HLS_L)
                        mean_lab_full_block = np.mean(lab_patch_block, axis=(0,1))
                        norm_a_star = mean_lab_full_block[1] / (mean_lab_full_block[0] + 1e-6)
                        norm_b_star = mean_lab_full_block[2] / (mean_lab_full_block[0] + 1e-6)
                        color_invariants = get_color_invariants(rgb_patch)
                        std_hls_l = np.std(patch_HLS_L)
                        var_hls_l = np.var(patch_HLS_L)
                        std_lab_l = np.std(patch_L_star)
                        var_lab_l = np.var(patch_L_star)
                        
                        spectral_features = {
                            'mean_r': mean_rgb[0], 'mean_g': mean_rgb[1], 'mean_b': mean_rgb[2],
                            'mean_hls_l': mean_hls_l, 'mean_lab_l': mean_lab_l,
                            'norm_a_star': norm_a_star, 'norm_b_star': norm_b_star,
                            'std_hls_l': std_hls_l, 'var_hls_l': var_hls_l,
                            'std_lab_l': std_lab_l, 'var_lab_l': var_lab_l
                        }
                        spectral_features.update(color_invariants)
                        features.update(spectral_features)
                        # Set column names only on the first pass
                        if not spectral_cols: spectral_cols = list(spectral_features.keys())

                    if include_glcm:
                        glcm_features_hls = get_glcm_features(patch_HLS_L, 'hls_l')
                        glcm_features_lab = get_glcm_features(patch_L_star, 'lab_l')
                        features.update(glcm_features_hls)
                        features.update(glcm_features_lab)
                        if not glcm_cols: glcm_cols = list(glcm_features_hls.keys()) + list(glcm_features_lab.keys())

                    if include_lbp:
                        lbp_features_hls = get_lbp_histogram(patch_HLS_L, 'hls_l')
                        lbp_features_lab = get_lbp_histogram(patch_L_star, 'lab_l')
                        features.update(lbp_features_hls)
                        features.update(lbp_features_lab)
                        if not lbp_cols: lbp_cols = list(lbp_features_hls.keys()) + list(lbp_features_lab.keys())
                    
                    all_features.append(features)

    df = pd.DataFrame(all_features)
    
    # Define column order dynamically
    base_cols = ['label_id', 'block_row', 'block_col']
    column_order = base_cols + spectral_cols + glcm_cols + lbp_cols + ['class']
    
    # Filter DataFrame to only include selected columns (and base/class)
    # This handles cases where user deselects a feature family
    final_columns = [col for col in column_order if col in df.columns]
    
    # Handle empty dataframe case
    if df.empty:
        print("Warning: No valid blocks found. Returning empty DataFrame.")
        # Return an empty DF with the base columns to avoid errors downstream
        return pd.DataFrame(columns=base_cols + ['class'])
        
    final_df = df[final_columns]
    
    return final_df
