import os
import json
import atexit
from concurrent.futures import ProcessPoolExecutor
import joblib
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from .block_features import (
    MIN_VALID_FRACTION, grid_shape, grid_transform, read_rgb_window, block_fraction, features_for_blocks
)
from .postprocess import finalize, NODATA
from .roi_utils import load_optional_roi, rasterize_roi_mask


def load_model_meta(model_path):
    meta_path = os.path.join(os.path.dirname(model_path), 'final_model_meta.json')
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        return json.load(f)


def predict_proba_blocks(src, model, features, n_classes, block_size, row0, col0, n_rows, n_cols, block_mask=None):
    """
    Class probabilities for the blocks [row0:row0+n_rows, col0:col0+n_cols] of the global grid.

    Returns
    -------
    P : (n_classes, n_rows, n_cols) float32, NaN for blocks that were not classified
        (incomplete image data or outside `block_mask`).
    valid_fraction : (n_rows, n_cols) fraction of valid image pixels (0 outside `block_mask`).
    """
    rgb, valid = read_rgb_window(src, row0, col0, n_rows, n_cols, block_size)
    valid_fraction = block_fraction(valid, block_size)
    if block_mask is not None:
        valid_fraction = np.where(block_mask, valid_fraction, 0.0)
    block_ok = valid_fraction >= MIN_VALID_FRACTION
    P = np.full((n_classes, n_rows, n_cols), np.nan, dtype='float32')
    if block_ok.any():
        X = features_for_blocks(rgb, block_size, block_ok, features)
        proba = model.predict_proba(pd.DataFrame(X, columns=features))
        P[np.asarray(model.classes_)[:, None], np.nonzero(block_ok)[0], np.nonzero(block_ok)[1]] = proba.T
    return P, valid_fraction


def predict_blocks(src, model, features, block_size, row0, col0, n_rows, n_cols, block_mask=None):
    """Unsmoothed class labels for a block window (NODATA where not classified)."""
    n_classes = int(np.max(model.classes_)) + 1
    P, _ = predict_proba_blocks(src, model, features, n_classes, block_size, row0, col0, n_rows, n_cols, block_mask)
    labels, _ = finalize(P, np.zeros(P.shape[1:]), smoothing=1, fill_min_valid=None)
    return labels


# Every band is read exactly once, so GDAL's block cache (by default 5 % of RAM, per process)
# only costs memory: keep it small.
GDAL_CACHE_MB = 64

# Per-process state for parallel inference (set by _init_worker)
_worker = {}


def _close_worker():
    """Closes the worker's dataset and GDAL environment before the process exits."""
    if 'src' in _worker:
        _worker.pop('src').close()
    if 'env' in _worker:
        _worker.pop('env').__exit__(None, None, None)


def _init_worker(ortho_path, model_path, features, n_classes, block_size, chunk_cols, in_worker=True):
    if in_worker:
        _worker['env'] = rasterio.Env(GDAL_CACHEMAX=GDAL_CACHE_MB)   # lives as long as the worker process
        _worker['env'].__enter__()
        atexit.register(_close_worker)
    model = joblib.load(model_path)
    if hasattr(model, 'n_jobs'):
        model.n_jobs = 1                       # parallelism comes from the band workers
    _worker.update(src=rasterio.open(ortho_path), model=model, features=features, n_classes=n_classes,
                   block_size=block_size, chunk_cols=chunk_cols)


def _predict_band(row0, n_rows, col0, n_cols, block_mask):
    """
    Class probabilities of a band of blocks, read from disk in one piece (efficient for
    orthophotos stored as full-width strips) and classified in column chunks.
    """
    w = _worker
    M, chunk = w['block_size'], w['chunk_cols']
    rgb, valid = read_rgb_window(w['src'], row0, col0, n_rows, n_cols, M)
    valid_fraction = block_fraction(valid, M)
    if block_mask is not None:
        valid_fraction = np.where(block_mask, valid_fraction, 0.0)
    P = np.full((w['n_classes'], n_rows, n_cols), np.nan, dtype='float32')
    for c0 in range(0, n_cols, chunk):
        c1 = min(c0 + chunk, n_cols)
        block_ok = valid_fraction[:, c0:c1] >= MIN_VALID_FRACTION
        if not block_ok.any():
            continue
        X = features_for_blocks(rgb[:, c0 * M:c1 * M], M, block_ok, w['features'])
        proba = w['model'].predict_proba(pd.DataFrame(X, columns=w['features']))
        rr, cc = np.nonzero(block_ok)
        P[np.asarray(w['model'].classes_)[:, None], rr, cc + c0] = proba.T
    return row0, col0, P, valid_fraction


def run_inference(ortho_path, model_path, feat_path, class_map_path, output_raster, block_size=50, tile_size=300,
                  roi_path=None, roi_layer_name='roi', smoothing=1, fill_min_valid=0.5, write_probabilities=False,
                  n_jobs=1, max_band_mb=64):
    """
    Full-map inference.

    The orthophoto is processed in horizontal bands of block rows (about `max_band_mb` of image
    data each), classified in chunks of about tile_size x tile_size pixels (small chunks keep
    memory low at almost no cost in speed). With n_jobs > 1
    (or -1 for all cores but one) bands are processed in parallel worker processes; n_jobs=1
    runs in the current process.

    Writes the class map, a confidence raster (probability of the mapped class, in %) next to
    it, and optionally the per-class probabilities. `smoothing` and `fill_min_valid` are
    described in postprocess.finalize.
    """
    features = joblib.load(feat_path)
    class_map = joblib.load(class_map_path)
    n_classes = len(class_map)

    trained_block = load_model_meta(model_path).get('block_size')
    if trained_block is not None and trained_block != block_size:
        raise ValueError(f"Model was trained on {trained_block}px blocks but inference uses block_size={block_size}.")

    with rasterio.open(ortho_path) as src:
        out_h, out_w = grid_shape(src, block_size)
        transform = grid_transform(src, block_size)
        crs = src.crs
        roi_gdf = load_optional_roi(roi_path, roi_layer_name, target_crs=src.crs)
    roi_mask = rasterize_roi_mask(roi_gdf, (out_h, out_w), transform)
    roi_mask = None if roi_mask is None else roi_mask.astype(bool)

    # Bands: full width (or the ROI's column range), height limited by memory
    band_rows = max(1, int(max_band_mb * 2 ** 20 // (out_w * block_size * block_size * 4)))
    jobs = []
    for r0 in range(0, out_h, band_rows):
        nr = min(band_rows, out_h - r0)
        c0, nc, mask = 0, out_w, None
        if roi_mask is not None:
            cols = np.flatnonzero(roi_mask[r0:r0 + nr].any(axis=0))
            if len(cols) == 0:
                continue
            c0, nc = cols[0], cols[-1] + 1 - cols[0]
            mask = roi_mask[r0:r0 + nr, c0:c0 + nc]
        jobs.append((r0, nr, c0, nc, mask))
    chunk_cols = max(1, max(1, tile_size // block_size) ** 2 // band_rows)   # ~ tile_size^2 pixels per chunk

    P = np.full((n_classes, out_h, out_w), np.nan, dtype='float32')
    valid_fraction = np.zeros((out_h, out_w), dtype='float32')
    init_args = (ortho_path, model_path, features, n_classes, block_size, chunk_cols)

    if n_jobs == -1:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    with rasterio.Env(GDAL_CACHEMAX=GDAL_CACHE_MB):
        if n_jobs == 1:
            _init_worker(*init_args, in_worker=False)
            results = (_predict_band(*job) for job in jobs)
            pool = None
        else:
            # Scripts that call this must be guarded by `if __name__ == '__main__':` (worker processes
            # re-import the main module); notebooks and QGIS need nothing.
            pool = ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker, initargs=init_args)
            results = pool.map(_predict_band, *zip(*jobs)) if jobs else []
        try:
            for r0, c0, P_band, vf_band in tqdm(results, total=len(jobs), desc="Processing"):
                nr, nc = vf_band.shape
                P[:, r0:r0 + nr, c0:c0 + nc] = P_band
                valid_fraction[r0:r0 + nr, c0:c0 + nc] = vf_band
        finally:
            if pool is not None:
                pool.shutdown()
            else:
                _close_worker()

    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'count': 1,
        'nodata': NODATA,
        'height': out_h,
        'width': out_w,
        'transform': transform,
        'crs': crs,
        'compress': 'lzw',
        'tiled': True
    }

    labels, confidence = finalize(P, valid_fraction, smoothing, fill_min_valid)
    out_dir = os.path.dirname(output_raster)
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(labels, 1)
    print(f"Classification saved to {output_raster}")

    conf_path = os.path.splitext(output_raster)[0] + '_confidence.tif'
    conf = np.where(np.isnan(confidence), NODATA, np.round(np.nan_to_num(confidence) * 100)).astype('uint8')
    with rasterio.open(conf_path, 'w', **profile) as dst:
        dst.write(conf, 1)
        dst.set_band_description(1, 'probability of mapped class (%)')
    print(f"Confidence saved to {conf_path}")

    if write_probabilities:
        prob_path = os.path.splitext(output_raster)[0] + '_probabilities.tif'
        probs = np.where(np.isnan(P), NODATA, np.round(np.nan_to_num(P) * 100)).astype('uint8')
        with rasterio.open(prob_path, 'w', **{**profile, 'count': n_classes}) as dst:
            dst.write(probs)
            for k, name in sorted(class_map.items()):
                dst.set_band_description(k + 1, f'P({name}) %')
        print(f"Class probabilities saved to {prob_path}")

    # Export class dictionary for human-readable interpretation of pixel values
    dict_path = os.path.join(out_dir, 'class_dictionary.json')
    with open(dict_path, 'w') as f:
        json.dump({str(k): v for k, v in class_map.items()}, f, indent=2)
    print(f"Class dictionary saved to {dict_path}")
    print("\nClass Dictionary (pixel value → class name):")
    for k, v in sorted(class_map.items()):
        print(f"  {k:>3} → {v}")
