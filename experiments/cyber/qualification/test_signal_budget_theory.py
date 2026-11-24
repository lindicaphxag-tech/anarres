"""Tests for the signal budget law.

Run from this directory:  python3 -m unittest test_signal_budget_theory
Experiment-local scripts are not on the package `make test` discovery path, so
this suite is run manually and documented in the commit.
"""

from __future__ import annotations

import unittest

import signal_budget_theory as theory


class IdentityTest(unittest.TestCase):
    def test_expected_mass_matches_its_closed_form(self) -> None:
        for p in (0.005, 0.035, 0.1, 0.25, 0.5, 0.75):
            for group_size in (2, 3, 8, 16, 32):
                mean = theory.group_moments(p, group_size)[0]
                self.assertAlmostEqual(mean, 2 * group_size * p * (1 - p), places=12)

    def test_published_concentration_rebuilds(self) -> None:
        rebuilt = theory.published_check()
        self.assertEqual(rebuilt["H"], 0.2518)
        self.assertEqual(rebuilt["advantage_mass"], 27.43)
        self.assertEqual(rebuilt["coverage"], 13 / 50)

    def test_coverage_matches_the_released_sweep(self) -> None:
        p = 14 / 400
        predicted = theory.coverage(p, 8)
        self.assertAlmostEqual(predicted, 0.248, places=3)
        self.assertAlmostEqual(predicted, 0.26, delta=0.015)


class BudgetTest(unittest.TestCase):
    def test_sparse_precision_is_flat_in_depth(self) -> None:
        p = 14 / 400
        narrow = theory.pooled_relative_se(p, 2)
        wide = theory.pooled_relative_se(p, 32)
        self.assertGreater(wide / narrow, 0.94)
        self.assertLess(wide / narrow, 1.0)

    def test_dense_precision_improves_with_depth(self) -> None:
        narrow = theory.pooled_relative_se(0.5, 2)
        wide = theory.pooled_relative_se(0.5, 32)
        self.assertGreater(narrow / wide, 4.0)

    def test_coverage_targets_need_depth(self) -> None:
        p = 14 / 400
        self.assertEqual(theory.coverage_group_size(p, 0.25), 9)
        self.assertEqual(theory.coverage_group_size(p, 0.5), 20)
        self.assertEqual(theory.coverage_group_size(p, 0.75), 39)

    def test_published_interval_is_the_budget_floor(self) -> None:
        p = 14 / 400
        relative = theory.pooled_relative_se(p, 8)
        predicted = theory.PUBLISHED["rho"] * (1 + 1.96 * relative)
        self.assertAlmostEqual(predicted, 33.59, delta=0.6)
        self.assertAlmostEqual(theory.PUBLISHED["rho"] * (1 - 1.96 * relative), 11.25, delta=0.6)


class AttenuationTest(unittest.TestCase):
    def test_sparse_signal_loses_almost_the_whole_false_negative_rate(self) -> None:
        loss = 1 - theory.density_ratio(0.035, 0.38)
        self.assertAlmostEqual(loss, 0.38, delta=0.02)

    def test_half_success_rate_has_no_first_order_effect(self) -> None:
        self.assertAlmostEqual(theory.density_slope(0.5), 0.0, places=12)
        self.assertLess(theory.density_ratio(0.5, 0.38), 1.0)

    def test_mostly_succeeding_tasks_report_more_signal_under_a_bad_verifier(self) -> None:
        self.assertGreater(theory.density_ratio(0.75, 0.38), 1.3)

    def test_attenuation_is_monotone_in_the_false_negative_rate(self) -> None:
        ratios = [theory.density_ratio(0.035, rate) for rate in (0.0, 0.1, 0.38, 0.5)]
        self.assertEqual(ratios, sorted(ratios, reverse=True))


if __name__ == "__main__":
    unittest.main()
