import unittest

import torch

from boundary_fsor.models.boundary import (
    TaskConditionedBoundaryGenerator,
    boundary_margin,
    paired_margin_matrix,
)


class BoundaryTest(unittest.TestCase):
    def test_margin_shape_and_correspondence(self):
        torch.manual_seed(1)
        positive = torch.nn.functional.normalize(torch.randn(5, 16), dim=-1)
        model = TaskConditionedBoundaryGenerator(16, 8)
        boundary = model(positive)
        query = positive[[1, 3]]
        margin, scores, winners = boundary_margin(query, positive, boundary)
        self.assertEqual(tuple(margin.shape), (2,))
        self.assertEqual(tuple(scores.shape), (2, 5))
        self.assertEqual(winners.tolist(), [1, 3])

    def test_global_companion_is_shared(self):
        model = TaskConditionedBoundaryGenerator(16, 8, global_only=True)
        companions = model(torch.randn(5, 16))
        torch.testing.assert_close(companions, companions[:1].expand_as(companions))

    def test_known_margin_can_use_ground_truth_class(self):
        positive = torch.eye(3)
        negative = torch.nn.functional.normalize(
            torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
            dim=-1,
        )
        query = torch.tensor([[0.1, 0.9, 0.0]])
        margins = paired_margin_matrix(query, positive, negative)
        self.assertNotEqual(float(margins[0, 0]), float(margins[0, 1]))


if __name__ == "__main__":
    unittest.main()
