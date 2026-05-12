import os
import json
import joblib
import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.color import rgb2lab, rgb2hsv
from .feature_extractor_masterv2 import get_color_invariants, get_glcm_features, get_lbp_histogram
from .roi_utils import load_optional_roi, rasterize_roi_mask

def calculate_block_features(rgb_patch, required_features, calc_spectral, calc_glcm, calc_lbp):
    """Calculates features for a single block during inference."""
    feature_dict = {}
    if rgb_patch.shape[0] == 0 or rgb_patch.shape[1] == 0:
        return pd.Series(dtype='float64').reindex(required_features)

    lab_patch = rgb2lab(rgb_patch)
    hsv_patch = rgb2hsv(rgb_patch)
    patch_L_star = lab_patch[:, :, 0]
    patch_HLS_L = (hsv_patch[:, :, 2] * 255).astype(np.uint8)

    if calc_spectral:
        mean_rgb = np.mean(rgb_patch, axis=(0, 1))
        mean_lab = np.mean(lab_patch, axis=(0, 1))
        feature_dict.update({
            'mean_r': mean_rgb[0], 'mean_g': mean_rgb[1], 'mean_b': mean_rgb[2],
            'mean_hls_l': np.mean(patch_HLS_L), 'mean_lab_l': np.mean(patch_L_star),
            'norm_a_star': mean_lab[1] / (mean_lab[0] + 1e-6),
            'norm_b_star': mean_lab[2] / (mean_lab[0] + 1e-6),
            'std_hls_l': np.std(patch_HLS_L), 'var_hls_l': np.var(patch_HLS_L),
            'std_lab_l': np.std(patch_L_star), 'var_lab_l': np.var(patch_L_star)
        })
        feature_dict.update(get_color_invariants(rgb_patch))

    if calc_glcm:
        feature_dict.update(get_glcm_features(patch_HLS_L, 'hls_l'))
        feature_dict.update(get_glcm_features(patch_L_star, 'lab_l'))

    if calc_lbp:
        feature_dict.update(get_lbp_histogram(patch_HLS_L, 'hls_l'))
        feature_dict.update(get_lbp_histogram(patch_L_star, 'lab_l'))

    return pd.Series(feature_dict).reindex(required_features).fillna(0)

def run_inference(ortho_path, model_path, feat_path, class_map_path, output_raster, block_size=50, tile_size=1000,
                  roi_path=None, roi_layer_name='roi'):
    """Full tile-based inference logic."""
    model = joblib.load(model_path)
    features = joblib.load(feat_path)
    class_map = joblib.load(class_map_path)

    calc_spectral = any(f in features for f in ['mean_r', 'std_hls_l', 'c1_invariant'])
    calc_glcm = any(f.startswith('glcm_') for f in features)
    calc_lbp = any(f.startswith('lbp_') for f in features)

    with rasterio.open(ortho_path) as src:
        roi_gdf = load_optional_roi(roi_path, roi_layer_name, target_crs=src.crs)
        roi_mask = rasterize_roi_mask(roi_gdf, (src.height, src.width), src.transform)

        # Create a clean profile for the output
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'count': 1,
            'nodata': 255,
            'height': src.height // block_size,
            'width': src.width // block_size,
            'transform': src.transform * src.transform.scale(block_size, block_size),
            'crs': src.crs,
            'compress': 'lzw',
            'tiled': True
        }
        
        out_h, out_w = profile['height'], profile['width']
        raster = np.full((out_h, out_w), 255, dtype='uint8')
        
        windows = [Window(c, r, tile_size, tile_size) for r in range(0, src.height, tile_size) for c in range(0, src.width, tile_size)]
        
        for win in tqdm(windows, desc="Processing"):
            if roi_mask is not None:
                mask_window = roi_mask[int(win.row_off):int(win.row_off + win.height), int(win.col_off):int(win.col_off + win.width)]
                if not mask_window.any():
                    continue

            img = np.transpose(src.read((1,2,3), window=win, boundless=True, fill_value=0), (1,2,0))
            if not img.any(): continue
            
            chunk_feats, coords = [], []
            for r in range(0, img.shape[0], block_size):
                for c in range(0, img.shape[1], block_size):
                    out_r, out_c = (win.row_off + r) // block_size, (win.col_off + c) // block_size
                    if out_r >= out_h or out_c >= out_w: continue
                    if roi_mask is not None:
                        roi_block = roi_mask[int(win.row_off + r):int(win.row_off + r + block_size), int(win.col_off + c):int(win.col_off + c + block_size)]
                        if roi_block.size == 0 or not roi_block.any():
                            continue
                    block = img[r:r+block_size, c:c+block_size]
                    if block.shape != (block_size, block_size, 3) or not block.any(): continue
                    chunk_feats.append(calculate_block_features(block, features, calc_spectral, calc_glcm, calc_lbp))
                    coords.append((out_r, out_c))
            
            if chunk_feats:
                preds = model.predict(pd.DataFrame(chunk_feats)).astype('uint8')
                rows, cols = zip(*coords)
                raster[rows, cols] = preds

        if os.path.exists(output_raster):
            try:
                os.remove(output_raster)
            except OSError:
                pass

        with rasterio.open(output_raster, 'w', **profile) as dst:
            dst.write(raster, 1)
    print(f"Classification saved to {output_raster}")
    
    # Export class dictionary for human-readable interpretation of pixel values
    dict_path = os.path.join(os.path.dirname(output_raster), 'class_dictionary.json')
    with open(dict_path, 'w') as f:
        json.dump({str(k): v for k, v in class_map.items()}, f, indent=2)
    print(f"Class dictionary saved to {dict_path}")
    print("\nClass Dictionary (pixel value → class name):")
    for k, v in sorted(class_map.items()):
        print(f"  {k:>3} → {v}")
