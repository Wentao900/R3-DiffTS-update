import unittest

import torch

from utils.text_signals import VALIDITY_FEATURE_DIM
from utils.text_validity import POLICY_CONDITION_BASE_DIM, VALIDITY_CONDITION_EXTRA_DIM, TextValidityModule


class TextValidityModuleTests(unittest.TestCase):
    def test_gate_can_approach_zero_without_a_floor(self):
        module = TextValidityModule(text_dim=8, proj_dim=4, hidden_dim=8, evidence_dim=VALIDITY_FEATURE_DIM, stats_dim=8)
        with torch.no_grad():
            module.sample_head.bias.fill_(-20.0)
        batch = 4
        source = torch.randn(batch, 3, 8)
        mask = torch.tensor([[1.0, 0.0, 1.0]] * batch)
        stats = torch.rand(batch, 8)
        evidence = torch.randn(batch, VALIDITY_FEATURE_DIM)
        disagreement = torch.rand(batch, 2)
        sample_gate, source_gates = module(source, mask, stats, evidence, disagreement)
        self.assertTrue(torch.all(sample_gate < 0.05))
        self.assertTrue(torch.all(sample_gate > 0.0))
        self.assertTrue(torch.all(source_gates[:, 1] == 0.0))
        self.assertTrue(torch.all(source_gates[:, 0] > 0.0))
        self.assertEqual(POLICY_CONDITION_BASE_DIM + VALIDITY_CONDITION_EXTRA_DIM, 29)

    def test_missing_retrieval_does_not_zero_the_sample_gate(self):
        module = TextValidityModule(text_dim=8, proj_dim=4, hidden_dim=8, evidence_dim=VALIDITY_FEATURE_DIM, stats_dim=8)
        source = torch.randn(2, 3, 8)
        mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        sample_gate, source_gates = module(
            source,
            mask,
            torch.rand(2, 8),
            torch.randn(2, VALIDITY_FEATURE_DIM),
            torch.zeros(2, 2),
        )
        self.assertTrue(torch.all(sample_gate > 0.2))
        self.assertTrue(torch.all(source_gates[:, 1] == 0.0))


if __name__ == "__main__":
    unittest.main()
