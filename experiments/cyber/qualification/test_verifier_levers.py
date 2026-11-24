"""Tests for the verifier lever pricing.

Run from this directory:  python3 -m unittest test_verifier_levers
"""

from __future__ import annotations

import unittest

import verifier_levers as levers


class EffectiveRateTest(unittest.TestCase):
    def test_clean_verifier_passes_the_rate_through(self) -> None:
        self.assertAlmostEqual(levers.effective_rate(0.035), 0.035, places=12)

    def test_total_false_negatives_zero_the_rate(self) -> None:
        self.assertAlmostEqual(levers.effective_rate(0.035, false_negative=1.0), 0.0, places=12)

    def test_false_positives_raise_the_rate(self) -> None:
        self.assertGreater(levers.effective_rate(0.035, false_positive=0.1), 0.035)


class LeverPriceTest(unittest.TestCase):
    def test_cutting_false_negatives_is_worth_about_thirty_percent_of_rate(self) -> None:
        price = levers.lever_price(0.035, 0.38, 0.20)
        self.assertAlmostEqual(price["recovered_density"], 1.282, delta=0.01)
        self.assertAlmostEqual(price["base_rate_equivalent"], 1.29, delta=0.02)


class ContaminationTest(unittest.TestCase):
    def test_a_ten_percent_false_positive_rate_misranks_the_panel(self) -> None:
        row = levers.contamination(0.035, 0.10)
        self.assertEqual(row["misranks"], 1.0)
        self.assertGreater(row["dirty_over_rival"], 1.4)

    def test_two_percent_false_positives_create_a_phantom_one_and_a_half(self) -> None:
        self.assertAlmostEqual(
            levers.observed_density(0.035, false_positive=0.02) / levers.density(0.035),
            1.52,
            delta=0.01,
        )


class AttributionTest(unittest.TestCase):
    def test_two_programmes_produce_the_same_observation(self) -> None:
        row = levers.ambiguity()
        self.assertAlmostEqual(row["rate_bad_verifier"], 0.035, places=9)
        self.assertAlmostEqual(row["rate_clean_verifier"], 0.0217, delta=0.001)
        self.assertGreater(row["gap"], 0.3)

    def test_audit_cost_stays_below_the_panel(self) -> None:
        self.assertGreater(levers.audit_items(0.38, 0.03), 900)
        self.assertLess(levers.audit_items(0.38, 0.03), 1200)
        self.assertLess(levers.audit_items(0.02, 0.02), 300)


if __name__ == "__main__":
    unittest.main()
