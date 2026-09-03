import unittest

import torch

from boundary_fsor.training.uncertainty import predictive_uncertainty


class UncertaintyTest(unittest.TestCase):
    def test_disagreement_increases_uncertainty(self):
        stable = torch.tensor([[[0.9, 0.1], [0.9, 0.1]]])
        disagree = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
        self.assertGreater(
            float(predictive_uncertainty(disagree)),
            float(predictive_uncertainty(stable)),
        )

    def test_probability_renormalization_after_class_slice(self):
        full = torch.tensor([[[0.4, 0.4, 0.2], [0.3, 0.6, 0.1]]])
        sliced = full[..., :2]
        sliced = sliced / sliced.sum(-1, keepdim=True)
        torch.testing.assert_close(sliced.sum(-1), torch.ones(1, 2))


if __name__ == "__main__":
    unittest.main()
