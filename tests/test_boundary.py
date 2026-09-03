import unittest

import torch

from boundary_fsor.models.boundary import BoundaryCompanion, boundary_margin


class BoundaryTest(unittest.TestCase):
    def test_margin_shape_and_correspondence(self):
        torch.manual_seed(1)
        positive = torch.nn.functional.normalize(torch.randn(5, 16), dim=-1)
        model = BoundaryCompanion(16, 8)
        boundary = model(positive)
        query = positive[[1, 3]]
        margin, scores, winners = boundary_margin(query, positive, boundary)
        self.assertEqual(tuple(margin.shape), (2,))
        self.assertEqual(tuple(scores.shape), (2, 5))
        self.assertEqual(winners.tolist(), [1, 3])

    def test_global_companion_is_shared(self):
        model = BoundaryCompanion(16, 8, global_only=True)
        companions = model(torch.randn(5, 16))
        torch.testing.assert_close(companions, companions[:1].expand_as(companions))


if __name__ == "__main__":
    unittest.main()
