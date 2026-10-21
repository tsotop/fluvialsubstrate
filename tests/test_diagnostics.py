"""
Diagnostics: tips are produced for typical annotation and training problems.

Run from the repository root:  python -m unittest discover tests
"""
import unittest

from src.diagnostics import _pair_advice, annotation_tips, training_tips


def polygon(label_id, cls, x, blocks_used, blocks_touched=None, y=0.0):
    return {'label_id': label_id, 'class': cls, 'area_m2': 10.0, 'x': x, 'y': y, 'block_area_m2': 1.0,
            'blocks_used': blocks_used, 'blocks_touched': blocks_touched or blocks_used}


class DiagnosticsTest(unittest.TestCase):

    def test_pair_advice_types(self):
        self.assertIn('moisture', _pair_advice('wet-sand', 'dry-sand', 1.0)[0])
        self.assertIn('grain sizes', _pair_advice('dry-cobble-small', 'dry-cobble-large', 1.0)[0])
        self.assertIn('moisture and grain size', _pair_advice('dry-cobble', 'wet-boulder', 1.0)[0])
        self.assertIn('Bedrock', _pair_advice('wet-bedrock', 'wet-cobble', 1.0)[0])
        self.assertIn('Vegetation', _pair_advice('dry-vegetation', 'wet-cobble', 1.0)[0])

    def test_annotation_problems(self):
        polys = [polygon(i, 'gravel', x=100.0 * i, blocks_used=40) for i in range(6)]
        polys += [polygon(10, 'boulder', x=10, blocks_used=30, blocks_touched=120),
                  polygon(11, 'boulder', x=20, blocks_used=0, blocks_touched=5)]
        tips = annotation_tips({'polygons': polys})
        text = ' '.join(t.message for t in tips if 'boulder' in t.classes)
        self.assertIn('only 1 usable polygon', text)
        self.assertIn('small or narrow', text)
        self.assertIn('give no training block', text)
        self.assertIn('lie within about', text)
        self.assertFalse([t for t in tips if 'gravel' in t.classes and t.severity == 'high'])

    def test_training_problems(self):
        cv = {'classes': ['a', 'b'],
              'spatial_cv': {'macro_f1': 0.5, 'confusion': [[20, 80], [5, 95]],
                             'report': {'a': {'recall': 0.2}, 'b': {'recall': 0.95}},
                             'polygons': [{'label_id': 1, 'class': 'b', 'n_blocks': 10, 'agreement': 0.1,
                                           'most_predicted': 'a', 'most_predicted_share': 0.9},
                                          {'label_id': 2, 'class': 'b', 'n_blocks': 90, 'agreement': 1.0,
                                           'most_predicted': 'b', 'most_predicted_share': 1.0}]},
              'standard_cv': {'macro_f1': 0.95}}
        messages = ' '.join(t.message for t in training_tips(cv, 1.0))
        self.assertIn("'a' is recognised in only 20%", messages)
        self.assertIn("are confused", messages)
        self.assertIn("Polygon 1", messages)
        self.assertIn("on unseen polygons", messages)


if __name__ == '__main__':
    unittest.main()
