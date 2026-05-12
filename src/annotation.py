import geopandas
import numpy as np
from shapely.geometry import Point
from collections import OrderedDict
import rasterio
import os

def generate_sampling_grid(ortho_path, output_path, grid_rows=5, grid_cols=5, points_per_cell=5):
    """
    Generates points within a virtual grid, an empty annotations layer,
    and an optional empty ROI layer.
    """
    print(f"Generating sampling grid for {ortho_path}...")
    
    try:
        with rasterio.open(ortho_path) as src:
            minx, miny, maxx, maxy = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
            ortho_crs = src.crs
            
            if not ortho_crs:
                raise ValueError("CRS not found in orthophoto.")
    except Exception as e:
        print(f"Error reading orthophoto: {e}")
        return False

    if os.path.exists(output_path):
        os.remove(output_path)

    cell_width = (maxx - minx) / grid_cols
    cell_height = (maxy - miny) / grid_rows

    all_points = []
    cell_id_counter = 0

    for i in range(grid_cols):
        for j in range(grid_rows):
            cell_minx = minx + i * cell_width
            cell_maxx = cell_minx + cell_width
            cell_miny = miny + j * cell_height
            cell_maxy = cell_miny + cell_height

            for _ in range(points_per_cell):
                rand_x = np.random.uniform(cell_minx, cell_maxx)
                rand_y = np.random.uniform(cell_miny, cell_maxy)
                point = Point(rand_x, rand_y)

                all_points.append({
                    'point_id': f"{cell_id_counter}_{_}",
                    'cell_id': cell_id_counter,
                    'geometry': point
                })
            cell_id_counter += 1

    points_gdf = geopandas.GeoDataFrame(all_points, crs=ortho_crs)
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
