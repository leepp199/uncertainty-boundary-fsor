import unittest

import numpy as np

from boundary_fsor.training.confusability import (
    ConfusabilityBank,
    ces_sampling_probabilities,
    linear_beta,
)


class ConfusabilityTest(unittest.TestCase):
    def setUp(self):
        embeddings = np.asarray([
            [1.0, 0.0], [0.9, 0.1],
            [0.8, 0.2], [0.7, 0.3],
            [0.0, 1.0], [0.1, 0.9],
        ])
        labels = np.asarray([0, 0, 1, 1, 2, 2])
        self.bank = ConfusabilityBank.from_embeddings(
            embeddings, labels, {0: 0.1, 1: 0.9, 2: 0.4},
        )

    def test_beta_reverses_confusability_preference(self):
        scores = np.asarray([0.1, 0.4, 0.9])
        easy = ces_sampling_probabilities(scores, beta=-2.0, epsilon=0.1)
        hard = ces_sampling_probabilities(scores, beta=2.0, epsilon=0.1)
        self.assertGreater(easy[0], easy[-1])
        self.assertGreater(hard[-1], hard[0])
        self.assertAlmostEqual(float(easy.sum()), 1.0)
        self.assertAlmostEqual(float(hard.sum()), 1.0)

    def test_beta_schedule_reaches_final_value(self):
        self.assertEqual(linear_beta(0, 1500, -2.0, 2.0), -2.0)
        self.assertEqual(linear_beta(750, 1500, -2.0, 2.0), 0.0)
        self.assertEqual(linear_beta(6000, 1500, -2.0, 2.0), 2.0)

    def test_distributions_exclude_known_classes(self):
        known = self.bank.known_distribution(beta=2.0, epsilon=0.1)
        unknown = self.bank.unknown_distribution(
            [0], [1, 2], beta=2.0, epsilon=0.1,
        )
        self.assertAlmostEqual(sum(known.values()), 1.0)
        self.assertAlmostEqual(sum(unknown.values()), 1.0)
        self.assertNotIn(0, unknown)

    def test_roundtrip_preserves_statistics(self):
        restored = ConfusabilityBank.from_state_dict(self.bank.state_dict())
        np.testing.assert_array_equal(restored.classes, self.bank.classes)
        np.testing.assert_allclose(restored.centers, self.bank.centers)
        np.testing.assert_allclose(restored.similarity, self.bank.similarity)


if __name__ == "__main__":
    unittest.main()
