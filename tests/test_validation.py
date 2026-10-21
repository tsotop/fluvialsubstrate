"""
Design-based estimators reproduce the worked example of Olofsson et al. (2014), Remote Sensing of
Environment 148:42-57, Section 5 (Tables 8-9).

Run from the repository root:  python -m unittest discover tests
"""
import unittest

import numpy as np

from src.validation import design_based_estimates

CLASSES = ['deforestation', 'forest gain', 'stable forest', 'stable non-forest']
COUNTS = np.array([[66, 0, 5, 4], [0, 55, 8, 12], [1, 0, 153, 11], [2, 1, 9, 313]])   # rows map, cols reference
AREA_PX = np.array([200000, 150000, 3200000, 6450000])                                 # mapped area (30 m pixels)


def sample_units(map_classes=None):
    W = AREA_PX / AREA_PX.sum()
    stratum, weight, mapped, reference = [], [], [], []
    for i in range(4):
        for j in range(4):
            n = COUNTS[i, j]
            stratum += [i] * n
            weight += [W[i]] * n
            mapped += [(map_classes or CLASSES)[i]] * n
            reference += [CLASSES[j]] * n
    return stratum, weight, mapped, reference


class OlofssonExampleTest(unittest.TestCase):

    def test_published_estimates(self):
        est = design_based_estimates(*sample_units(), CLASSES)
        z = 1.96
        self.assertAlmostEqual(est['overall']['accuracy'], 0.947, places=3)
        self.assertAlmostEqual(z * est['overall']['se'], 0.018, places=3)
        pc = est['per_class'].set_index('class')
        np.testing.assert_allclose(pc['user_accuracy'], [0.88, 0.73, 0.93, 0.96], atol=0.005)
        np.testing.assert_allclose(pc['producer_accuracy'][['deforestation', 'stable forest']], [0.75, 0.93], atol=0.005)
        total_ha = AREA_PX.sum() * 0.09
        self.assertAlmostEqual(pc.loc['deforestation', 'area_proportion'] * total_ha, 21158, delta=1)
        self.assertAlmostEqual(z * pc.loc['deforestation', 'area_proportion_se'] * total_ha, 6158, delta=2)
        self.assertAlmostEqual(est['error_matrix'].to_numpy().sum(), 1.0)

    def test_map_equal_to_strata_gives_mapped_proportions(self):
        est = design_based_estimates(*sample_units(), CLASSES)
        np.testing.assert_allclose(est['per_class']['mapped_proportion'], AREA_PX / AREA_PX.sum())


if __name__ == '__main__':
    unittest.main()
