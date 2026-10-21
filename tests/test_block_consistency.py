"""
Consistency tests for the block grid shared by training and inference.

Run from the repository root:  python -m unittest discover tests
"""
import os
import tempfile
import unittest

import geopandas as gpd
import joblib
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from src.block_features import ALL_FEATURES, FEATURE_FAMILIES, read_rgb_window, block_fraction, features_for_blocks, grid_shape
from src.features import extract_block_features
from src.inference import predict_blocks, run_inference
from src.roi_utils import load_optional_roi
from src.annotation import seed_points
from src.postprocess import finalize, NODATA

M = 20
PIX = 0.1
X0, Y0 = 500000.0, 5100000.0


class ThresholdModel:
    """Deterministic stand-in classifier: class depends on the first feature."""
    classes_ = np.arange(7)

    def predict(self, X):
        return np.clip((np.asarray(X)[:, 0] // 40).astype(int), 0, 6)

    def predict_proba(self, X):
        P = np.zeros((len(X), 7))
        P[np.arange(len(X)), self.predict(X)] = 1.0
        return P


def write_ortho(path, height=230, width=310, alpha=True):
    rng = np.random.default_rng(0)
    rgb = rng.integers(1, 255, size=(3, height, width), dtype=np.uint8)
    rgb[:, :, :40] = 0                       # black collar (no-data without metadata)
    rgb[:, :, 280:] = 255                    # white collar (the other common fill colour)
    count = 4 if alpha else 3
    with rasterio.open(path, 'w', driver='GTiff', height=height, width=width, count=count, dtype='uint8',
                       crs='EPSG:32632', transform=from_origin(X0, Y0, PIX, PIX)) as dst:
        dst.write(rgb, (1, 2, 3))
        if alpha:
            a = np.full((height, width), 255, np.uint8)
            a[200:, :] = 0                   # transparent strip at the bottom
            dst.write(a, 4)
            dst.colorinterp = [rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
                               rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha]


def pixel_box(r0, c0, r1, c1):
    """Polygon covering pixel rows r0..r1 and cols c0..c1 (end exclusive)."""
    return box(X0 + c0 * PIX, Y0 - r1 * PIX, X0 + c1 * PIX, Y0 - r0 * PIX)


class BlockConsistencyTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ortho = os.path.join(cls.tmp.name, 'ortho.tif')
        write_ortho(cls.ortho)
        cls.labels = os.path.join(cls.tmp.name, 'labels.gpkg')
        gdf = gpd.GeoDataFrame({
            'substrate': ['a', 'b', 'c'],
            'geometry': [
                pixel_box(25, 65, 117, 151),     # misaligned with the grid: partial edge blocks
                pixel_box(150, 30, 230, 120),    # touches black collar and transparent strip
                pixel_box(40, 200, 100, 280),
            ]}, crs='EPSG:32632')
        gdf.to_file(cls.labels, layer='annotations', driver='GPKG')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def extract(self, coverage):
        return extract_block_features(M, self.labels, 'annotations', self.ortho, families=tuple(FEATURE_FAMILIES),
                                      min_label_coverage=coverage, roi_path=None)

    def test_training_matches_inference_features(self):
        df = self.extract(0.5)
        self.assertFalse(df.empty)
        with rasterio.open(self.ortho) as src:
            n_rows, n_cols = grid_shape(src, M)
            # Whole-image window: different window origin than any polygon window
            rgb, valid = read_rgb_window(src, 0, 0, n_rows, n_cols, M)
            ok = np.zeros((n_rows, n_cols), bool)
            ok[df.block_row, df.block_col] = True
            X = features_for_blocks(rgb, M, ok, ALL_FEATURES)
        rows, cols = np.nonzero(ok)
        ref = df.set_index(['block_row', 'block_col']).loc[list(zip(rows, cols)), ALL_FEATURES].to_numpy()
        np.testing.assert_allclose(X, ref, rtol=1e-10, atol=1e-10)

    def test_coverage_threshold(self):
        full = self.extract(1.0)
        loose = self.extract(0.01)
        self.assertTrue((full.label_coverage == 1.0).all())
        self.assertGreater(len(loose), len(full))
        # Polygon 'a' spans pixel rows 25-117, cols 65-151 -> complete blocks rows 2-4, cols 4-6
        a = full[full['class'] == 'a']
        self.assertEqual(set(a.block_row), {2, 3, 4})
        self.assertEqual(set(a.block_col), {4, 5, 6})

    def test_invalid_pixels_excluded(self):
        df = self.extract(0.01)
        self.assertFalse((df.block_col < 2).any(), "blocks in the black collar must be dropped")
        self.assertFalse((df.block_row >= 10).any(), "blocks in the transparent strip must be dropped")

    def test_white_fill_excluded(self):
        df = self.extract(0.01)
        self.assertFalse((df.block_col >= 14).any(), "blocks in the white collar must be dropped")

    def test_seed_points_on_valid_pixels(self):
        pts = seed_points(self.ortho, seed=1)
        self.assertGreater(len(pts), 0)
        with rasterio.open(self.ortho) as src:
            for g in pts.geometry:
                r, c = src.index(g.x, g.y)
                _, valid = read_rgb_window(src, r, c, 1, 1, 1)
                self.assertTrue(valid[0, 0])

    def test_smoothing_and_fill(self):
        P = np.full((2, 5, 5), np.nan)
        P[:, :, :4] = np.array([0.9, 0.1])[:, None, None]       # class 0 everywhere ...
        P[:, 2, 2] = [0.4, 0.6]                                 # ... except one noisy block
        frac = np.ones((5, 5))
        frac[:, 4] = [0.6, 0.6, 0.2, 0.0, 0.6]                  # partly valid edge column
        raw, _ = finalize(P, frac, smoothing=1, fill_min_valid=None)
        self.assertEqual(raw[2, 2], 1)
        self.assertTrue((raw[:, 4] == NODATA).all())
        smooth, conf = finalize(P, frac, smoothing=3, fill_min_valid=0.5)
        self.assertEqual(smooth[2, 2], 0, "isolated block is smoothed into its neighbourhood")
        self.assertEqual(list(smooth[:, 4]), [0, 0, NODATA, NODATA, 0], "only blocks >= 50% valid are filled")
        self.assertTrue(np.isnan(conf[2, 4]))

    def test_missing_roi_layer_means_no_roi(self):
        gpd.GeoDataFrame({'geometry': [pixel_box(0, 0, 1, 1).centroid]}, crs='EPSG:32632').to_file(
            self.labels, layer='points', driver='GPKG')
        self.assertIsNone(load_optional_roi(self.labels, 'roi'))

    def test_inference_independent_of_tile_size(self):
        model_dir = os.path.join(self.tmp.name, 'model')
        os.makedirs(model_dir, exist_ok=True)
        paths = {k: os.path.join(model_dir, f'{k}.joblib') for k in ('model', 'feats', 'classes')}
        joblib.dump(ThresholdModel(), paths['model'])
        joblib.dump(['mean_r', 'glcm_hsv_v_contrast'], paths['feats'])
        joblib.dump({0: 'x', 1: 'y', 2: 'z', 3: 'w', 4: 'v', 5: 'u', 6: 't'}, paths['classes'])

        maps = []
        for tile in (70, 50, 10_000):     # 70 is not a multiple of M
            out = os.path.join(self.tmp.name, f'map_{tile}.tif')
            run_inference(self.ortho, paths['model'], paths['feats'], paths['classes'], out,
                          block_size=M, tile_size=tile, smoothing=3)
            with rasterio.open(out) as src:
                maps.append(src.read(1))
        out = os.path.join(self.tmp.name, 'map_parallel.tif')
        run_inference(self.ortho, paths['model'], paths['feats'], paths['classes'], out,
                      block_size=M, tile_size=70, smoothing=3, n_jobs=2, max_band_mb=0.001)
        with rasterio.open(out) as src:
            maps.append(src.read(1))
        for m in maps[1:]:
            np.testing.assert_array_equal(maps[0], m)
        self.assertEqual(maps[0].shape, (230 // M, 310 // M))
        self.assertTrue((maps[0][:, :2] == 255).all(), "collar blocks must be nodata")
        self.assertTrue((maps[0][10:, :] == 255).all(), "transparent blocks must be nodata")
        self.assertTrue((maps[0][:, 14:] == 255).all(), "white collar blocks must be nodata")
        self.assertTrue((maps[0][:10, 2:14] != 255).all())


if __name__ == '__main__':
    unittest.main()
