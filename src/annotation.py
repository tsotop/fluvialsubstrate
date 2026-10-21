import geopandas
import numpy as np
from collections import OrderedDict
import rasterio
from rasterio.features import rasterize
import os
from .block_features import valid_overview
from .roi_utils import load_optional_roi


def seed_points(ortho_path, grid_rows=5, grid_cols=5, points_per_cell=5, seed=42, roi_gdf=None, max_dim=2000):
    """
    Stratified seed points inside the valid image area (and the ROI, if given).

    The grid spans the bounding box of the valid area; each cell receives up to
    `points_per_cell` points drawn uniformly from its valid part. Cells without
    valid pixels receive none.
    """
    rng = np.random.default_rng(seed)
    with rasterio.open(ortho_path) as src:
        if not src.crs:
            raise ValueError("CRS not found in orthophoto.")
        crs = src.crs
        valid, transform = valid_overview(src, max_dim)
        if roi_gdf is not None:
            roi_gdf = roi_gdf.to_crs(crs)
            valid &= rasterize(((g, 1) for g in roi_gdf.geometry), out_shape=valid.shape,
                               transform=transform, fill=0, dtype='uint8').astype(bool)

    rows, cols = np.nonzero(valid)
    if len(rows) == 0:
        raise ValueError("No valid image area found (check the ROI and the orthophoto's no-data).")

    # Grid over the bounding box of the valid area, in overview pixel coordinates
    r_edges = np.linspace(rows.min(), rows.max() + 1, grid_rows + 1)
    c_edges = np.linspace(cols.min(), cols.max() + 1, grid_cols + 1)
    cell_r = np.clip(np.searchsorted(r_edges, rows + 0.5, side='right') - 1, 0, grid_rows - 1)
    cell_c = np.clip(np.searchsorted(c_edges, cols + 0.5, side='right') - 1, 0, grid_cols - 1)

    points = []
    for i in range(grid_cols):
        for j in range(grid_rows):
            cell_id = i * grid_rows + j
            idx = np.nonzero((cell_c == i) & (cell_r == j))[0]
            if len(idx) == 0:
                continue
            pick = rng.choice(idx, size=min(points_per_cell, len(idx)), replace=False)
            for k, p in enumerate(pick):
                # Uniform position inside the chosen overview pixel
                x, y = transform * (cols[p] + rng.random(), rows[p] + rng.random())
                points.append({'point_id': f"{cell_id}_{k}", 'cell_id': cell_id,
                               'geometry': geopandas.points_from_xy([x], [y])[0]})

    n_cells = len({p['cell_id'] for p in points})
    print(f"{len(points)} seed points in {n_cells} of {grid_rows * grid_cols} grid cells "
          f"(cells without valid image{' or ROI' if roi_gdf is not None else ''} area are skipped).")
    return geopandas.GeoDataFrame(points, columns=['point_id', 'cell_id', 'geometry'], geometry='geometry', crs=crs)


def generate_sampling_grid(ortho_path, output_path, grid_rows=5, grid_cols=5, points_per_cell=5,
                           seed=42, overwrite=False, points_only=False):
    """
    Generates seed points within a virtual grid, an empty annotations layer,
    and an optional empty ROI layer.

    Points are placed only on valid image pixels (not on the mosaic's no-data fill).

    - An existing GeoPackage is never replaced unless `overwrite=True`, so re-running
      this step cannot erase annotations.
    - `points_only=True` regenerates only the 'points' layer of an existing GeoPackage,
      restricted to the ROI drawn in its 'roi' layer; annotations and ROI are kept.
    """
    if points_only:
        if not os.path.exists(output_path):
            print(f"⚠️ {output_path} does not exist; run without points_only first.")
            return False
        roi_gdf = load_optional_roi(output_path, 'roi')
        print(f"Regenerating seed points for {ortho_path}"
              f"{' inside the ROI' if roi_gdf is not None else ' (no ROI drawn: whole valid area)'}...")
        points_gdf = seed_points(ortho_path, grid_rows, grid_cols, points_per_cell, seed, roi_gdf)
        points_gdf.to_file(output_path, layer='points', driver='GPKG')
        print(f"Updated 'points' layer in {output_path}")
        return True

    if os.path.exists(output_path) and not overwrite:
        print(f"⚠️ {output_path} already exists; keeping it (and any annotations). "
              "Pass points_only=True to regenerate only the seed points, or overwrite=True to start over.")
        return False

    print(f"Generating sampling grid for {ortho_path}...")
    try:
        points_gdf = seed_points(ortho_path, grid_rows, grid_cols, points_per_cell, seed)
    except Exception as e:
        print(f"Error reading orthophoto: {e}")
        return False
    ortho_crs = points_gdf.crs

    if os.path.exists(output_path):
        os.remove(output_path)

    points_gdf.to_file(output_path, layer='points', driver='GPKG')

    annotations_schema = {
        'geometry': 'Polygon',
        'properties': OrderedDict([('substrate', 'str:50')])
    }
    annotations_gdf = geopandas.GeoDataFrame(
        columns=['substrate', 'geometry'],
        geometry='geometry',
        crs=ortho_crs
    )
    annotations_gdf.to_file(
        output_path,
        layer='annotations',
        driver='GPKG',
        schema=annotations_schema,
        engine='fiona'
    )

    roi_schema = {
        'geometry': 'Polygon',
        'properties': OrderedDict([('roi_id', 'str:50')])
    }
    roi_gdf = geopandas.GeoDataFrame(columns=['roi_id', 'geometry'], geometry='geometry', crs=ortho_crs)
    roi_gdf.to_file(
        output_path,
        layer='roi',
        driver='GPKG',
        schema=roi_schema,
        engine='fiona'
    )

    print(f"Saved to {output_path}")
    return True
