import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from scipy.stats import norm
import os
import json
import pandas as pd
from .roi_utils import load_optional_roi, rasterize_roi_mask

def generate_validation_points(map_path, class_map_path, output_dir, total_samples=150, min_per_class=20, square_size_px=1,
                               roi_path=None, roi_layer_name='roi', seed=42):
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
        seed (int): Random seed, so the same map always yields the same sample.
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

    rng = np.random.default_rng(seed)
    samples_list = []
    for cls, count in zip(unique_classes, counts):
        wh = count / total_pixels
        n_h = max(int(wh * total_samples), min_per_class)
        rows, cols = np.where((data == cls) & (roi_mask == 1))
        if len(rows) < n_h: n_h = len(rows)
        idx = rng.choice(len(rows), n_h, replace=False)
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

    full_gdf = gpd.GeoDataFrame(samples_list, crs=crs).sample(frac=1, random_state=seed).reset_index(drop=True)
    
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

    n_suggested, _ = suggest_sample_size(map_path, min_per_class=min_per_class, roi_path=roi_path,
                                         roi_layer_name=roi_layer_name)
    print(f"Note: for a standard error of overall accuracy of about 2 % (assuming 75 % user's accuracy per "
          f"class), about {n_suggested} samples are needed (Olofsson et al. 2014); see suggest_sample_size().")
    print(f"Saved {len(annot_gdf)} validation squares to:")
    print(f"  → {annot_path}  (fill 'true_label' in QGIS)")
    print(f"  → {model_path}  (model predictions — keep hidden during annotation)")
    
    # Print class dictionary for reference
    print("\nClass Dictionary (pixel value → class name):")
    for k, v in sorted(class_map.items()):
        print(f"  {k:>3} → {v}")


def _stratified_mean(df, value):
    """Stratified estimate of the population mean of column `value` and its standard error."""
    est, var = 0.0, 0.0
    for _, h in df.groupby('stratum'):
        w, n = h['W'].iloc[0], len(h)
        est += w * h[value].mean()
        if n > 1:
            var += w ** 2 * h[value].var(ddof=1) / n
    return est, np.sqrt(var)


def _stratified_ratio(df, num, den):
    """Combined ratio estimator R = sum W_h ybar_h / sum W_h xbar_h and its standard error."""
    Y, _ = _stratified_mean(df, num)
    X, _ = _stratified_mean(df, den)
    if X == 0:
        return np.nan, np.nan
    R = Y / X
    var = 0.0
    for _, h in df.groupby('stratum'):
        w, n = h['W'].iloc[0], len(h)
        if n > 1:
            c = np.cov(h[num], h[den], ddof=1)
            var += w ** 2 * (c[0, 0] + R ** 2 * c[1, 1] - 2 * R * c[0, 1]) / n
    return R, np.sqrt(max(var, 0.0)) / X


def design_based_estimates(stratum, weight, mapped, reference, classes):
    """
    Accuracy and area estimates from a stratified random sample (Stehman 2014; Olofsson et al. 2014).

    Parameters
    ----------
    stratum : stratum of every sample unit (normally the map class used to draw the sample).
    weight : W_h, the share of the mapped population in the unit's stratum.
    mapped, reference : class of the evaluated map and the reference (true) class at every unit.
        The evaluated map may differ from the map that defined the strata (e.g. a re-trained map
        scored on an existing sample); the estimators remain unbiased.
    classes : class names to report.

    Returns a dict with 'overall' (accuracy, se), 'per_class' (DataFrame: user's and producer's
    accuracy, F1, mapped and estimated area proportions, with standard errors) and
    'error_matrix' (DataFrame of estimated area proportions, rows = map, columns = reference).
    """
    df = pd.DataFrame({'stratum': np.asarray(stratum), 'W': np.asarray(weight, dtype=float),
                       'map': np.asarray(mapped), 'ref': np.asarray(reference)})
    df['correct'] = (df['map'] == df['ref']).astype(float)
    oa, oa_se = _stratified_mean(df, 'correct')

    rows, matrix = [], {}
    for k in classes:
        df['x'] = (df['map'] == k).astype(float)
        df['y'] = (df['ref'] == k).astype(float)
        df['xy'] = df['x'] * df['y']
        ua, ua_se = _stratified_ratio(df, 'xy', 'x')
        pa, pa_se = _stratified_ratio(df, 'xy', 'y')
        mapped_p, _ = _stratified_mean(df, 'x')
        area_p, area_se = _stratified_mean(df, 'y')
        f1 = 2 * ua * pa / (ua + pa) if (ua + pa) > 0 else 0.0
        rows.append({'class': k, 'n_mapped': int(df['x'].sum()), 'n_reference': int(df['y'].sum()),
                     'user_accuracy': ua, 'user_accuracy_se': ua_se,
                     'producer_accuracy': pa, 'producer_accuracy_se': pa_se, 'f1': f1,
                     'mapped_proportion': mapped_p, 'area_proportion': area_p, 'area_proportion_se': area_se})
        matrix[k] = [_stratified_mean(df.assign(z=((df['map'] == i) & (df['ref'] == k)).astype(float)), 'z')[0]
                     for i in classes]
    error_matrix = pd.DataFrame(matrix, index=list(classes))
    error_matrix.index.name, error_matrix.columns.name = 'map', 'reference'
    return {'overall': {'accuracy': oa, 'se': oa_se, 'n': len(df)},
            'per_class': pd.DataFrame(rows), 'error_matrix': error_matrix}


def suggest_sample_size(map_path, target_se=0.02, expected_user_accuracy=0.75, min_per_class=20,
                        roi_path=None, roi_layer_name='roi'):
    """
    Sample size for a target standard error of overall accuracy (Olofsson et al. 2014, eq. 13),
    n = (sum_h W_h S_h / target_se)^2 with S_h = sqrt(U_h (1 - U_h)), and its allocation
    (proportional to mapped area, at least `min_per_class` per class).
    `expected_user_accuracy` is one value or a dict class value -> expected user's accuracy.
    """
    with rasterio.open(map_path) as src:
        data, transform, crs, nodata = src.read(1), src.transform, src.crs, src.nodata
    roi_mask = rasterize_roi_mask(load_optional_roi(roi_path, roi_layer_name, target_crs=crs), data.shape, transform)
    valid = (data != nodata) if roi_mask is None else (data != nodata) & (roi_mask == 1)
    classes, counts = np.unique(data[valid], return_counts=True)
    W = counts / counts.sum()
    U = np.array([expected_user_accuracy.get(int(c), 0.75) if isinstance(expected_user_accuracy, dict)
                  else expected_user_accuracy for c in classes])
    n = int(np.ceil((np.sum(W * np.sqrt(U * (1 - U))) / target_se) ** 2))
    allocation = {int(c): max(int(round(w * n)), min_per_class) for c, w in zip(classes, W)}
    return n, allocation


def _resolve_names(values, int_to_name):
    """Reference labels may be typed as class names or as the map's integer codes."""
    out = []
    for v in values:
        v = str(v).strip()
        try:
            out.append(int_to_name.get(int(float(v)), v))
        except ValueError:
            out.append(v)
    return out


def calculate_accuracy(annotations_path, modeled_path, output_dir, confidence=0.95, map_path=None,
                       class_map_path=None, show_plot=True):
    """
    Design-based accuracy and area assessment of a classified map from a labelled validation sample.

    The sample is the one written by `generate_validation_points` (true labels in
    `annotations_path`, strata and weights in `modeled_path`; both may be the same file).
    By default the map that defined the strata is assessed. With `map_path` another map of the
    same area (e.g. after re-training) is assessed on the same sample, without new labelling.

    Writes accuracy_summary.csv, accuracy_per_class.csv, error_matrix_area.csv and a confusion
    matrix figure to `output_dir`, and returns the estimates (see design_based_estimates).
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import joblib

    z = norm.ppf(1 - (1 - confidence) / 2)
    labels = gpd.read_file(annotations_path)
    strata = gpd.read_file(modeled_path)
    for g in (labels, strata):
        if 'sample_id' not in g.columns:
            g['sample_id'] = g.index
    labels = labels[labels['true_label'].notnull() & (labels['true_label'].astype(str).str.strip() != '')]
    n_skipped = len(strata) - len(labels)
    if labels.empty:
        print("Error: No annotated samples found in the validation annotations file.")
        return None
    if n_skipped > 0:
        print(f"Warning: {n_skipped} unannotated samples skipped. Proceeding with {len(labels)} annotated samples.")
    merged = strata.drop(columns=['true_label'], errors='ignore').merge(
        labels[['sample_id', 'true_label']], on='sample_id')

    # Class names: from the class mapping if given (or next to the map), else from the sample file
    int_to_name = {}
    if class_map_path is None and map_path is not None:
        candidate = os.path.join(os.path.dirname(map_path), 'class_dictionary.json')
        if os.path.exists(candidate):
            with open(candidate) as f:
                int_to_name = {int(k): v for k, v in json.load(f).items()}
    if class_map_path is not None:
        int_to_name = joblib.load(class_map_path)
    if not int_to_name and 'map_prediction_name' in merged.columns:
        int_to_name = merged.drop_duplicates('map_prediction').set_index('map_prediction')['map_prediction_name'].to_dict()
    if not int_to_name:
        raise ValueError("Class names unknown: pass class_map_path.")
    classes = [int_to_name[k] for k in sorted(int_to_name)]

    if map_path is None:
        mapped = [int_to_name.get(int(v), str(v)) for v in merged['map_prediction']]
        pixel_area = merged.geometry.area.iloc[0] if merged.geom_type.iloc[0] == 'Polygon' else np.nan
    else:
        with rasterio.open(map_path) as src:
            pts = merged.geometry.to_crs(src.crs).centroid
            values = [v[0] for v in src.sample([(p.x, p.y) for p in pts])]
            mapped = [int_to_name.get(int(v), 'no data') if v != src.nodata else 'no data' for v in values]
            pixel_area = abs(src.res[0] * src.res[1])
        if 'no data' in mapped:
            print(f"Warning: {mapped.count('no data')} samples fall on no-data in {map_path}; counted as errors.")
        # Strata keep the pixel size of the map that defined them
        if merged.geom_type.iloc[0] == 'Polygon':
            pixel_area = merged.geometry.area.iloc[0]

    reference = _resolve_names(merged['true_label'], int_to_name)
    unknown = sorted(set(reference) - set(classes))
    if unknown:
        print(f"⚠️ Warning: reference labels not matching any map class (counted as errors; check for typos): {unknown}")

    est = design_based_estimates(merged['map_prediction'], merged['stratum_weight'], mapped, reference, classes)
    per_class = est['per_class']
    total_area_ha = np.nan
    if 'stratum_area_pixels' in merged.columns and np.isfinite(pixel_area):
        total_area_ha = merged.drop_duplicates('map_prediction')['stratum_area_pixels'].sum() * pixel_area / 1e4
        per_class['mapped_area_ha'] = per_class['mapped_proportion'] * total_area_ha
        per_class['area_ha'] = per_class['area_proportion'] * total_area_ha
        per_class['area_ha_ci'] = z * per_class['area_proportion_se'] * total_area_ha

    # --- Report ---
    oa = est['overall']
    pct = lambda v, se: f"{100 * v:5.1f} ± {100 * z * se:4.1f}" if np.isfinite(v) else "    –"
    print("\n" + "=" * 96)
    print(f"  DESIGN-BASED ACCURACY ASSESSMENT ({len(merged)} samples, {int(confidence * 100)}% confidence intervals)")
    print("=" * 96)
    print(f"  Overall accuracy: {pct(oa['accuracy'], oa['se'])} %")
    if np.isfinite(total_area_ha):
        print(f"  Mapped area: {total_area_ha:.2f} ha")
    print("-" * 96)
    header = f"  {'Class':<22}{'User acc. %':>15}{'Prod. acc. %':>15}{'F1':>7}"
    if 'area_ha' in per_class:
        header += f"{'Mapped ha':>12}{'Estimated ha':>20}"
    print(header)
    for _, r in per_class.iterrows():
        line = (f"  {r['class']:<22}{pct(r['user_accuracy'], r['user_accuracy_se']):>15}"
                f"{pct(r['producer_accuracy'], r['producer_accuracy_se']):>15}{r['f1']:>7.2f}")
        if 'area_ha' in per_class:
            line += f"{r['mapped_area_ha']:>12.2f}{r['area_ha']:>12.2f} ± {r['area_ha_ci']:<6.2f}"
        print(line)
    print("=" * 96)
    print("User's accuracy: share of the area mapped as a class that truly is that class.")
    print("Producer's accuracy: share of the true area of a class that the map labels correctly.")
    print("Estimated area: area of each class corrected for the map errors found in the sample.")

    # --- Files ---
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame([{'overall_accuracy': oa['accuracy'], 'se': oa['se'], 'ci_low': oa['accuracy'] - z * oa['se'],
                   'ci_high': oa['accuracy'] + z * oa['se'], 'n_samples': oa['n'],
                   'mapped_area_ha': total_area_ha}]).to_csv(os.path.join(output_dir, 'accuracy_summary.csv'), index=False)
    per_class.to_csv(os.path.join(output_dir, 'accuracy_per_class.csv'), index=False)
    est['error_matrix'].to_csv(os.path.join(output_dir, 'error_matrix_area.csv'))

    present = [c for c in classes if c in set(reference) | set(mapped)]
    cm = confusion_matrix(reference, mapped, labels=present)
    fig, ax = plt.subplots(figsize=(max(6, len(present)), max(5, len(present) - 1)))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=present).plot(
        ax=ax, colorbar=False, cmap='Blues', xticks_rotation='vertical')
    ax.set_xlabel('Map'), ax.set_ylabel('Reference')
    ax.set_title("Validation sample (counts)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close(fig)
    print(f"\nResults saved to {output_dir}")

    est['total_area_ha'] = total_area_ha
    return est
