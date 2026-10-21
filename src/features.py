import os
import json
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from tqdm import tqdm
from .block_features import (
    ALL_FEATURES, FEATURE_FAMILIES, DEFAULT_FAMILIES, MIN_VALID_FRACTION,
    grid_shape, grid_transform, read_rgb_window, block_fraction, features_for_blocks
)
from .roi_utils import load_optional_roi
from .ml_utils import plot_class_distribution

FEATURE_VERSION = 3


def check_for_typos(gpkg_path, layer_name, threshold=85):
    """Checks for typos in the substrate labels."""
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
        if gdf.empty: return
        unique_classes = sorted(gdf['substrate'].dropna().unique())
        for i in range(len(unique_classes)):
            for j in range(i + 1, len(unique_classes)):
                ratio = 100 * SequenceMatcher(None, unique_classes[i].lower(), unique_classes[j].lower()).ratio()
                if ratio >= threshold:
                    print(f"⚠️ Warning: Potential typo '{unique_classes[i]}' <--> '{unique_classes[j]}'")
    except Exception as e:
        print(f"Typo check failed: {e}")


def extract_block_features(block_size, labels_path, layer_name, orthophoto_path, families=DEFAULT_FAMILIES,
                           min_label_coverage=0.9, roi_path=None, roi_layer_name='roi'):
    """
    Extracts features for every grid block covered by a labeled polygon.

    A block is kept when at least `min_label_coverage` of its pixels fall inside
    the polygon (and the ROI) and all of its pixels hold valid image data.
    Features are computed on the full block, exactly as during inference.
    """
    feature_names = [f for fam in families for f in FEATURE_FAMILIES[fam]]

    print(f"Reading labels from: {labels_path} (Layer: {layer_name})")
    try:
        labels = gpd.read_file(labels_path, layer=layer_name)
    except Exception as e:
        print(f"--- ERROR ---\nCould not read layer '{layer_name}' from file '{labels_path}'.\nError: {e}")
        return pd.DataFrame()

    labels = labels[labels.geometry.notnull() & ~labels.geometry.is_empty & labels['substrate'].notnull()]
    if labels.empty:
        print(f"Warning: No features found in layer '{layer_name}'. Returning empty DataFrame.")
        return pd.DataFrame()

    tables = []
    with rasterio.open(orthophoto_path) as src:
        if labels.crs != src.crs:
            print(f"Reprojecting labels from {labels.crs} to {src.crs}...")
            labels = labels.to_crs(src.crs)

        roi_gdf = load_optional_roi(roi_path or labels_path, roi_layer_name, target_crs=src.crs)
        roi_union = None if roi_gdf is None else roi_gdf.geometry.union_all()

        n_rows, n_cols = grid_shape(src, block_size)
        gtransform = grid_transform(src, block_size)
        n_dropped_cover = 0
        polygons = []          # per-polygon statistics, used by diagnostics
        block_area = abs(src.res[0] * src.res[1]) * block_size ** 2

        for row in tqdm(labels.itertuples(), total=len(labels), desc="Extracting block features"):
            geom = row.geometry if roi_union is None else row.geometry.intersection(roi_union)
            stats = {'label_id': int(row.Index), 'class': row.substrate, 'area_m2': float(geom.area),
                     'x': float(row.geometry.centroid.x), 'y': float(row.geometry.centroid.y),
                     'blocks_touched': 0, 'blocks_used': 0, 'block_area_m2': block_area}
            polygons.append(stats)
            if geom.is_empty:
                continue

            # Block range touched by the polygon, on the global grid
            win = from_bounds(*geom.bounds, transform=gtransform)
            r0, c0 = max(int(np.floor(win.row_off)), 0), max(int(np.floor(win.col_off)), 0)
            r1 = min(int(np.ceil(win.row_off + win.height)), n_rows)
            c1 = min(int(np.ceil(win.col_off + win.width)), n_cols)
            if r1 <= r0 or c1 <= c0:
                print(f"Skipping geometry {row.Index}: outside the orthophoto.")
                continue

            rgb, valid = read_rgb_window(src, r0, c0, r1 - r0, c1 - c0, block_size)
            pix_transform = src.window_transform(rasterio.windows.Window(
                c0 * block_size, r0 * block_size, (c1 - c0) * block_size, (r1 - r0) * block_size))
            inside = geometry_mask([geom], out_shape=valid.shape, transform=pix_transform, invert=True)

            coverage = block_fraction(inside, block_size)
            touched = coverage > 0
            block_ok = (coverage >= min_label_coverage) & (block_fraction(valid, block_size) >= MIN_VALID_FRACTION)
            n_dropped_cover += int(touched.sum() - block_ok.sum())
            stats.update(blocks_touched=int(touched.sum()), blocks_used=int(block_ok.sum()))
            if not block_ok.any():
                continue

            X = features_for_blocks(rgb, block_size, block_ok, feature_names)
            br, bc = np.nonzero(block_ok)
            t = pd.DataFrame(X, columns=feature_names)
            t.insert(0, 'label_id', row.Index)
            t.insert(1, 'block_row', br + r0)
            t.insert(2, 'block_col', bc + c0)
            t.insert(3, 'label_coverage', coverage[block_ok])
            t['class'] = row.substrate
            tables.append(t)

    if not tables:
        print("Warning: No valid blocks found. Returning empty DataFrame.")
        return pd.DataFrame(columns=['label_id', 'block_row', 'block_col', 'class'])

    df = pd.concat(tables, ignore_index=True)
    print(f"Blocks touched by polygons but dropped (coverage < {min_label_coverage} or invalid pixels): {n_dropped_cover}")
    empty = [p['label_id'] for p in polygons if p['blocks_used'] == 0]
    if empty:
        print(f"⚠️ {len(empty)} polygons gave no training block (too small or narrow for {block_size} px blocks): "
              f"label_ids {empty}")
    df = _resolve_overlaps(df)
    df.attrs['polygons'] = polygons
    return df


def _resolve_overlaps(df):
    """Blocks claimed by several polygons: keep one if labels agree, drop all if they conflict."""
    key = ['block_row', 'block_col']
    n_classes = df.groupby(key)['class'].transform('nunique')
    conflicts = df[n_classes > 1]
    if not conflicts.empty:
        n = conflicts[key].drop_duplicates().shape[0]
        ids = sorted(conflicts['label_id'].unique())
        print(f"⚠️ {n} blocks fall in overlapping polygons with different labels (label_ids {ids}); dropped.")
    df = df[n_classes == 1]
    df = df.sort_values('label_coverage', ascending=False).drop_duplicates(key).sort_index()
    return df.reset_index(drop=True)


def run_extraction(ortho_path, labels_path, layer_name, output_csv, block_size=50, min_label_coverage=0.9,
                   roi_path=None, roi_layer_name='roi', families=DEFAULT_FAMILIES, show_plot=True):
    """Wraps the feature extraction process."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    check_for_typos(labels_path, layer_name)

    print(f"Extracting features from {ortho_path}...")
    data_df = extract_block_features(
        block_size=block_size,
        labels_path=labels_path,
        layer_name=layer_name,
        orthophoto_path=ortho_path,
        families=families,
        min_label_coverage=min_label_coverage,
        roi_path=roi_path,
        roi_layer_name=roi_layer_name
    )

    if data_df.empty:
        print("Extraction returned no data.")
        return None

    data_df.to_csv(output_csv, index=False)
    meta = {
        'feature_version': FEATURE_VERSION,
        'block_size': block_size,
        'min_label_coverage': min_label_coverage,
        'families': list(families),
        'orthophoto': os.path.abspath(ortho_path),
        'features': [c for c in ALL_FEATURES if c in data_df.columns],
        'polygons': data_df.attrs.get('polygons', []),
    }
    with open(feature_meta_path(output_csv), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Extracted {len(data_df)} samples to {output_csv}")

    if show_plot:
        plot_class_distribution(data_df['class'])
    return data_df


def feature_meta_path(features_csv):
    return os.path.splitext(features_csv)[0] + '_meta.json'
