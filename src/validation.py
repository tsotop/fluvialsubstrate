import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from scipy.stats import norm
import os
import json
from .roi_utils import load_optional_roi, rasterize_roi_mask

def generate_validation_points(map_path, class_map_path, output_dir, total_samples=150, min_per_class=20, square_size_px=1,
                               roi_path=None, roi_layer_name='roi'):
    """
    Generates a stratified validation sample from the classified map.

    Writes two separate GeoPackage files:
    - 'validation_true_annotations.gpkg': geometry + sample_id + empty 'true_label' for user annotation.
    - 'validation_modeled.gpkg': geometry + sample_id + map prediction and stratum weights (hidden during annotation).

    Args:
        map_path (str): Path to the classified raster.
        class_map_path (str): Path to the class mapping joblib (int -> class name).
        output_dir (str): Directory where the two output files are written.
        total_samples (int): Total number of validation samples.
        min_per_class (int): Minimum samples per class.
        square_size_px (int): Ignored; squares are fixed to 1 classified pixel (= 1 model block).
        roi_path (str): Optional ROI file or GeoPackage containing an ROI layer.
        roi_layer_name (str): ROI layer name inside the ROI file.
    """
    import joblib
    class_map = joblib.load(class_map_path)
    
    print(f"Generating validation sample from {map_path}...")
    with rasterio.open(map_path) as src:
        data = src.read(1)
        transform, crs, nodata = src.transform, src.crs, src.nodata

    roi_gdf = load_optional_roi(roi_path, roi_layer_name, target_crs=crs)
    roi_mask = rasterize_roi_mask(roi_gdf, data.shape, transform)
    if roi_mask is None:
        roi_mask = np.ones(data.shape, dtype='uint8')

    valid_pixels = data[(data != nodata) & (roi_mask == 1)]
    unique_classes, counts = np.unique(valid_pixels, return_counts=True)
    total_pixels = counts.sum()
    
    # The classified map is block-resolution (1 pixel = 1 model block), so each
    # validation square is fixed to exactly one classified pixel.
    square_size_px = 1

    # Convert pixel size to map units so squares are MxM pixels in map coordinates.
    pixel_size_x = abs(transform.a)
    pixel_size_y = abs(transform.e)
    half_w = (square_size_px * pixel_size_x) / 2.0
    half_h = (square_size_px * pixel_size_y) / 2.0

    samples_list = []
    for cls, count in zip(unique_classes, counts):
        wh = count / total_pixels
        n_h = max(int(wh * total_samples), min_per_class)
        rows, cols = np.where((data == cls) & (roi_mask == 1))
        if len(rows) < n_h: n_h = len(rows)
        idx = np.random.choice(len(rows), n_h, replace=False)
        class_name = class_map.get(int(cls), f"class_{cls}")
        for i in idx:
            x, y = rasterio.transform.xy(transform, rows[i], cols[i], offset='center')
            samples_list.append({
                'geometry': box(x - half_w, y - half_h, x + half_w, y + half_h),
                'map_prediction': int(cls),
                'map_prediction_name': class_name,
                'true_label': None,
                'stratum_weight': wh,
                'stratum_area_pixels': int(count)
            })

    full_gdf = gpd.GeoDataFrame(samples_list, crs=crs).sample(frac=1).reset_index(drop=True)
    
    # File 1: Blind annotation — user sees only geometry + sample_id + empty true_label
    annot_gdf = full_gdf[['geometry']].copy()
    annot_gdf['sample_id'] = annot_gdf.index
    annot_gdf['true_label'] = None
    annot_gdf['stroke'] = '#ff0000'
    annot_gdf['stroke-width'] = 1.0
    annot_gdf['fill'] = '#000000'
    annot_gdf['fill-opacity'] = 0.0

    # File 2: Model reference — geometry + sample_id + predictions and weights (hidden during annotation)
    model_gdf = full_gdf[['geometry', 'map_prediction', 'map_prediction_name', 'stratum_weight', 'stratum_area_pixels']].copy()
    model_gdf['sample_id'] = model_gdf.index

    os.makedirs(output_dir, exist_ok=True)
    annot_path = os.path.join(output_dir, 'validation_true_annotations.gpkg')
    model_path = os.path.join(output_dir, 'validation_modeled.gpkg')

    for p in (annot_path, model_path):
        if os.path.exists(p):
            os.remove(p)

    annot_gdf.to_file(annot_path, driver="GPKG")
    model_gdf.to_file(model_path, driver="GPKG")

    print(f"Saved {len(annot_gdf)} validation squares to:")
    print(f"  → {annot_path}  (fill 'true_label' in QGIS)")
    print(f"  → {model_path}  (model predictions — keep hidden during annotation)")
    
    # Print class dictionary for reference
    print("\nClass Dictionary (pixel value → class name):")
    for k, v in sorted(class_map.items()):
        print(f"  {k:>3} → {v}")


def calculate_accuracy(annotations_path, modeled_path, output_dir, confidence=0.95):
    """
    Calculates design-based inference accuracy from the two validation files.
    Merges validation_true_annotations.gpkg (user-filled true_label) with validation_modeled.gpkg.
    Outputs a detailed per-class and overall accuracy report.
    """
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import pandas as pd

    # Read the user-annotated file
    blind_gdf = gpd.read_file(annotations_path)
    
    # Drop unannotated samples instead of erroring
    n_total = len(blind_gdf)
    blind_gdf = blind_gdf[blind_gdf['true_label'].notnull()].copy()
    n_annotated = len(blind_gdf)
    if n_annotated == 0:
        print("Error: No annotated samples found in the validation annotations file.")
        return
    if n_annotated < n_total:
        print(f"Warning: {n_total - n_annotated} unannotated samples skipped. Proceeding with {n_annotated} annotated samples.")
    
    # Read the model reference file
    ref_gdf = gpd.read_file(modeled_path)
    
    # Merge on sample_id
    merged = blind_gdf[['sample_id', 'true_label']].merge(
        ref_gdf[['sample_id', 'map_prediction', 'map_prediction_name', 'stratum_weight', 'stratum_area_pixels']],
        on='sample_id'
    )
    
    # Build integer → class name lookup from reference metadata
    int_to_name = ref_gdf.drop_duplicates('map_prediction').set_index('map_prediction')['map_prediction_name'].to_dict()
    
    # Convert true_label: if user entered an integer (e.g. "2"), map it to class name
    def resolve_label(val):
        val = str(val).strip()
        try:
            return int_to_name.get(int(val), val)   # try integer lookup first
        except (ValueError, TypeError):
            return val                                # already a string class name
    
    merged['true_label_name'] = merged['true_label'].apply(resolve_label)
    merged['pred_name'] = merged['map_prediction_name'].astype(str).str.strip()
    
    # Canonical label ordering (sorted by class int, so names appear alphabetically by int key)
    all_class_names = [int_to_name[k] for k in sorted(int_to_name.keys())]
    
    # ─── 1. Design-based overall accuracy (Olofsson et al.) ────────────────────
    print("\n" + "="*70)
    print("  DESIGN-BASED ACCURACY ASSESSMENT (Stratified Random Sampling)")
    print("="*70)
    
    classes = sorted(int_to_name.keys())
    acc_est, var_est = 0.0, 0.0
    dbi_rows = []
    for cls in classes:
        s_data = merged[merged['map_prediction'] == cls]
        if len(s_data) == 0:
            continue
        nh = len(s_data)
        wh = s_data.iloc[0]['stratum_weight']
        cls_name = int_to_name[cls]
        nh_corr = (s_data['true_label_name'] == cls_name).sum()
        ph = nh_corr / nh
        acc_est += wh * ph
        if nh > 1:
            var_est += (wh**2) * (ph * (1-ph)) / (nh - 1)
        dbi_rows.append({
            'Class': cls_name,
            'Stratum Weight (Wh)': f"{wh:.4f}",
            'Sample (nh)': nh,
            'Correct': nh_corr,
            'User Accuracy (ph)': f"{ph*100:.1f}%"
        })

    se = np.sqrt(var_est)
    me = norm.ppf(1 - (1 - confidence) / 2) * se

    dbi_df = pd.DataFrame(dbi_rows)
    print(dbi_df.to_string(index=False))
    print("-"*70)
    print(f"  Overall Estimated Accuracy: {acc_est*100:.2f}% ± {me*100:.2f}%  ({int(confidence*100)}% CI)")
    print("="*70)

    # ─── 2. Standard Per-Class Classification Report ───────────────────────────
    print("\n" + "="*70)
    print("  PER-CLASS PRECISION / RECALL / F1-SCORE")
    print("="*70)
    y_true = merged['true_label_name'].tolist()
    y_pred = merged['pred_name'].tolist()
    # Use the canonical order, restrict to classes that actually appear
    present_labels = [n for n in all_class_names if n in set(y_true) | set(y_pred)]
    print(classification_report(y_true, y_pred, labels=present_labels, zero_division=0))

    # ─── 3. Confusion Matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    fig, ax = plt.subplots(figsize=(max(6, len(present_labels)), max(5, len(present_labels) - 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=present_labels)
    disp.plot(ax=ax, colorbar=False, cmap='Blues', xticks_rotation='vertical')
    ax.set_title("Validation Confusion Matrix", fontsize=13, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.show()
    print(f"\nConfusion matrix saved to {cm_path}")

    return dbi_rows, acc_est, me
