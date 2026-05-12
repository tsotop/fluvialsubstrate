import os

import geopandas as gpd
from rasterio.features import rasterize


def load_optional_roi(roi_path, layer_name=None, target_crs=None):
    if not roi_path or not os.path.exists(roi_path):
        return None

    try:
        if layer_name:
            roi_gdf = gpd.read_file(roi_path, layer=layer_name)
        else:
            roi_gdf = gpd.read_file(roi_path)
    except Exception:
        try:
            roi_gdf = gpd.read_file(roi_path)
        except Exception:
            return None

    if roi_gdf.empty or 'geometry' not in roi_gdf.columns:
        return None

    roi_gdf = roi_gdf[roi_gdf.geometry.notnull() & ~roi_gdf.geometry.is_empty].copy()
    if roi_gdf.empty:
        return None

    if target_crs is not None and roi_gdf.crs is not None and roi_gdf.crs != target_crs:
        roi_gdf = roi_gdf.to_crs(target_crs)

    return roi_gdf if not roi_gdf.empty else None


def rasterize_roi_mask(roi_gdf, out_shape, transform):
    if roi_gdf is None or roi_gdf.empty:
        return None

    shapes = ((geom, 1) for geom in roi_gdf.geometry if geom is not None and not geom.is_empty)
    return rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype='uint8',
        all_touched=False,
    )