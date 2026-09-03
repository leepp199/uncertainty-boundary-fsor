import unittest

import numpy as np

from boundary_fsor.training.curriculum import BoundaryRiskTable, curriculum_rank_weights


class BoundaryRiskTest(unittest.TestCase):
    def setUp(self):
        embeddings = np.asarray([
            [1.0, 0.0], [0.9, 0.1],
            [0.8, 0.2], [0.7, 0.3],
            [0.0, 1.0], [0.1, 0.9],
        ])
        labels = np.asarray([0, 0, 1, 1, 2, 2])
        self.table = BoundaryRiskTable.from_statistics(
            embeddings, labels, {0: 0.1, 1: 0.9, 2: 0.4},
        )

    def test_curriculum_reverses_emphasis(self):
        values = np.asarray([0.1, 0.4, 0.9])
        early = curriculum_rank_weights(values, 0.0)
        late = curriculum_rank_weights(values, 1.0)
        self.assertGreater(early[0], early[-1])
        self.assertGreater(late[-1], late[0])
        self.assertTrue(np.all(early > 0.0))
        self.assertTrue(np.all(late > 0.0))

    def test_probabilities_are_normalized_and_exclude_known(self):
        known = self.table.class_probability(1.0)
        intrusion = self.table.intrusion_probability([0], 1.0)
        self.assertAlmostEqual(sum(known.values()), 1.0)
        self.assertAlmostEqual(sum(intrusion.values()), 1.0)
        self.assertNotIn(0, intrusion)

    def test_roundtrip_preserves_statistics(self):
        restored = BoundaryRiskTable.from_state_dict(self.table.state_dict())
        np.testing.assert_array_equal(restored.classes, self.table.classes)
        np.testing.assert_allclose(restored.centers, self.table.centers)
        np.testing.assert_allclose(restored.similarity, self.table.similarity)


if __name__ == "__main__":
    unittest.main()
