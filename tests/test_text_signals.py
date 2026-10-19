import unittest

import numpy as np

from utils.text_signals import (
    VALIDITY_FEATURE_DIM,
    build_raw_validity_features,
    compose_priority_text,
    infer_text_trend_fields,
    numeric_endpoint_stats,
    order_recent_segments,
)
from utils.text_validity import benefit_gate_target


class TextSignalTests(unittest.TestCase):
    def test_substring_keywords_do_not_fire(self):
        rainfall = infer_text_trend_fields("Heavy rainfall is expected across the region.")
        supply = infer_text_trend_fields("The supply report arrived on schedule.")
        self.assertEqual(rainfall["direction"], "flat")
        self.assertEqual(supply["direction"], "flat")

    def test_whole_word_direction_still_matches(self):
        self.assertEqual(infer_text_trend_fields("Prices fall after the shock.")["direction"], "down")
        self.assertEqual(infer_text_trend_fields("Demand went up this month.")["direction"], "up")
        self.assertEqual(infer_text_trend_fields("An upward revision.")["direction"], "up")

    def test_cot_json_is_parsed_not_keyword_scanned(self):
        cot = (
            '{"direction":"down","strength":"weak","volatility":"low",'
            '"key_factors":"numeric history"}'
        )
        fields = infer_text_trend_fields(cot)
        self.assertEqual(fields["direction"], "down")
        self.assertEqual(fields["strength"], "weak")
        self.assertEqual(fields["volatility"], "low")

    def test_cross_column_slope_uses_target_column(self):
        # Shape (time, feature). Target column falls. The flattened last value
        # is the last covariate, which would have produced a positive slope.
        window = np.array(
            [
                [5.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.0, 0.0, 100.0],
            ],
            dtype=float,
        )
        stats = numeric_endpoint_stats(window, target_index=0)
        self.assertLess(stats["slope"], 0.0)
        flat = window.reshape(-1)
        flattened_slope = (flat[-1] - flat[0]) / (flat.size - 1)
        self.assertGreater(flattened_slope, 0.0)

    def test_composed_text_puts_hypothesis_before_raw(self):
        text = compose_priority_text(
            raw_text="OLD RAW REPORT " * 40,
            numeric_summary="last=1, trend is down",
            trend_hypothesis='{"direction":"down","strength":"weak","volatility":"low"}',
            retrieved=["retrieved fact about exports"],
        )
        self.assertLess(text.index("[TREND HYPOTHESIS]"), text.index("[NUMERICAL SUMMARY]"))
        self.assertLess(text.index("[NUMERICAL SUMMARY]"), text.index("[RETRIEVED EVIDENCE]"))
        self.assertLess(text.index("[RETRIEVED EVIDENCE]"), text.index("[RAW TEXT]"))

    def test_recent_segments_are_newest_first(self):
        desc = "Domain description."
        text = order_recent_segments(
            desc,
            [desc, "old report", "middle report", "newest report"],
            token_budget=20,
        )
        self.assertLess(text.index("newest report"), text.index("middle report"))
        self.assertLess(text.index("newest report"), text.index("old report"))

    def test_raw_validity_features_are_not_a_product_score(self):
        features = build_raw_validity_features(
            raw_text="Prices fall sharply.",
            ret_text="",
            cot_text='{"direction":"up","strength":"strong","volatility":"low","key_factors":"x"}',
            seq_x=np.linspace(0.0, 1.0, 8),
            recency_days=30.0,
            report_count=4,
            recent_report_count=1,
            target_index=0,
        )
        self.assertEqual(features.shape, (VALIDITY_FEATURE_DIM,))
        self.assertEqual(features[0], 1.0)  # raw available
        self.assertEqual(features[7], 0.0)  # retrieved unavailable
        self.assertEqual(features[14], 1.0)  # cot available
        self.assertEqual(features[18], 1.0)  # cot direction up
        self.assertEqual(features[20], 0.0)  # cot volatility ordinal low, not high
        self.assertEqual(features[24], 30.0)  # raw age, not exp(-age/14)

    def test_benefit_target_tracks_which_routing_is_better(self):
        import torch

        err_num = torch.tensor([2.0, 1.0, 0.0])
        err_text = torch.tensor([1.0, 3.0, 0.0])
        target = benefit_gate_target(err_num, err_text)
        self.assertGreater(float(target[0]), 0.5)
        self.assertLess(float(target[1]), 0.5)
        self.assertAlmostEqual(float(target[2]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
