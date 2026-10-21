"""
The vectorised block features must equal the per-block skimage reference implementation.

Run from the repository root:  python -m unittest discover tests
"""
import unittest

import numpy as np
from skimage.color import rgb2lab

from src.block_features import ALL_FEATURES, _needed, features_for_blocks
from tests.reference_features import compute_block_features

M = 30


def test_window(seed=0):
    """A 3 x 4 block window with varied content: noise, gradients, constant, saturated, two-tone."""
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(3 * M, 4 * M, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:M, 0:M]
    rgb[:M, M:2 * M] = np.stack([xx * 8, yy * 8, (xx + yy) * 4], axis=-1).astype(np.uint8)   # smooth
    rgb[:M, 2 * M:3 * M] = (90, 120, 60)                                                    # constant
    rgb[:M, 3 * M:] = 255                                                                   # saturated
    rgb[M:2 * M, :M] = np.where((xx // 3 + yy // 3)[..., None] % 2, 200, 40).astype(np.uint8)  # checker
    rgb[2 * M:, :M] = rng.integers(100, 110, size=(M, M, 3), dtype=np.uint8)               # low contrast
    return rgb


class FeatureEquivalenceTest(unittest.TestCase):

    def test_vectorised_matches_reference(self):
        rgb = test_window()
        ok = np.ones((3, 4), bool)
        ok[2, 3] = False
        X = features_for_blocks(rgb, M, ok, ALL_FEATURES)

        lab = rgb2lab(rgb)
        channels = {
            'hsv_v': rgb.max(axis=2),
            'lab_l': lab[:, :, 0].astype(np.uint8),
            'lab_l8': np.clip(np.round(lab[:, :, 0] * 2.55), 0, 255).astype(np.uint8),
        }
        needed = _needed(ALL_FEATURES)
        for i, (r, c) in enumerate(zip(*np.nonzero(ok))):
            sl = (slice(r * M, (r + 1) * M), slice(c * M, (c + 1) * M))
            ref = compute_block_features(rgb[sl], lab[sl], {k: v[sl] for k, v in channels.items()}, needed)
            ref = np.nan_to_num(np.array([ref[n] for n in ALL_FEATURES], dtype=float))
            # LBP may differ from skimage in exact ties of the interpolated neighbours (see
            # lbp_uniform): allow a few pixels per histogram bin
            atol = np.array([4.0 / M ** 2 if '_bin_' in n else 1e-9 for n in ALL_FEATURES])
            bad = ~np.isclose(X[i], ref, rtol=1e-9, atol=atol)
            self.assertFalse(bad.any(), f"block ({r},{c}) differs: "
                             f"{[(n, X[i][k], ref[k]) for k, n in enumerate(ALL_FEATURES) if bad[k]][:5]}")


if __name__ == '__main__':
    unittest.main()
