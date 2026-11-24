"""Exact budget theory for advantage-mass measurement under a fixed rollout budget.

The question is what a fixed rollout budget buys. This module stays in closed
form on the binomial group law: reported numbers come from exact summation over
k = 0..G, and seeded simulation only checks the algebra.

Notation. A task has success probability p. G rollouts are drawn, k succeed.
The binary leave-one-out unnormalized advantage of rollout i is (G - k)/(G - 1)
on success and -k/(G - 1) on failure, so a group carries
A(k) = 2 k (G - k) / (G - 1) of advantage mass. Signal density rho is total
advantage mass per generated token. Effective coverage H is the normalized
concentration of that mass across tasks.

Three identities hold for every p and G:

1. E[A] = 2 G p (1 - p). The expectation carries no finite-G bias.
2. Var(A) has the closed form below and grows linearly in G when p G is small,
   and saturates at 1/2 when p = 1/2.
3. A group shows signal exactly when 0 < k < G, with probability
   1 - p^G - (1 - p)^G. At sparse p that probability is close to G p.

The second identity is what decides a design. Under a fixed budget B, the pooled
rho estimate pools B / G groups, so Var(rho_hat) = Var(A) / (G * B). Because
Var(A) is roughly linear in G in the sparse regime, that product is flat: at
p = 0.035 the same budget gives a relative standard error of 0.25 whether the
panel uses 2, 8, or 32 rollouts per task. Precision then does not constrain G,
and coverage alone does: the design picks G for the coverage it wants.

In the dense regime the product falls as 1/G, so the same budget buys much
sharper rho at larger G.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from math import comb
from pathlib import Path
from typing import Any

# Tokens per rollout in the released 50-task sweep: 1,223,456 output tokens over
# 400 rollouts.
TOKENS_PER_ROLLOUT = 1_223_456 / 400
BUDGET_ROLLOUTS = 400
PUBLISHED = {
    "tasks": 50,
    "rollouts_per_task": 8,
    "success_rollouts": 14,
    "coverage": 13 / 50,
    "rho": 22.42,
    "H": 0.2518,
    "advantage_mass": 27.43,
}


def advantage_mass(k: int, group_size: int) -> float:
    """Advantage mass of one group with k successes in group_size rollouts."""

    return 2 * k * (group_size - k) / (group_size - 1)


def group_moments(p: float, group_size: int) -> tuple[float, float]:
    """Exact mean and variance of the group advantage mass under Binomial(G, p)."""

    mean = 0.0
    second = 0.0
    for k in range(group_size + 1):
        weight = comb(group_size, k) * p**k * (1 - p) ** (group_size - k)
        value = advantage_mass(k, group_size)
        mean += weight * value
        second += weight * value * value
    return mean, second - mean * mean


def coverage(p: float, group_size: int) -> float:
    """Probability that a group is mixed, so its advantage mass is nonzero."""

    return 1 - p**group_size - (1 - p) ** group_size


def pooled_relative_se(p: float, group_size: int, budget: int = BUDGET_ROLLOUTS) -> float:
    """Relative standard error of the pooled rho estimate at a fixed budget."""

    groups = budget / group_size
    variance = groups * group_moments(p, group_size)[1] / budget**2
    expected = groups * group_moments(p, group_size)[0] / budget
    return variance**0.5 / expected


def coverage_group_size(p: float, target: float) -> int:
    """Smallest G whose mixed-group probability reaches target."""

    smallest = 1
    while coverage(p, smallest) < target:
        smallest += 1
    return smallest


def herfindahl(advantages: list[float]) -> float:
    """Normalized concentration of advantage mass over tasks."""

    total = sum(advantages)
    if total == 0:
        return 0.0
    return total * total / (len(advantages) * sum(value * value for value in advantages))


def monte_carlo_moments(p: float, group_size: int, draws: int = 200_000) -> tuple[float, float]:
    """Seeded check of the exact moments, used as a self-test rather than a result."""

    generator = random.Random(1_000_003 + group_size)
    values = []
    for _ in range(draws):
        k = 0
        for _ in range(group_size):
            k += 1 if generator.random() < p else 0
        values.append(advantage_mass(k, group_size))
    mean = sum(values) / draws
    variance = sum((value - mean) ** 2 for value in values) / (draws - 1)
    return mean, variance


def published_check() -> dict[str, Any]:
    """Rebuild the published concentration index from the published k vector."""

    advantages = [2.00] * 12 + [3.43]
    advantages += [0.0] * (PUBLISHED["tasks"] - len(advantages))
    return {
        "advantage_mass": round(sum(advantages), 2),
        "H": round(herfindahl(advantages), 4),
        "coverage": len([value for value in advantages if value > 0]) / PUBLISHED["tasks"],
    }


def shifted_probability(p: float, false_negative: float, false_positive: float = 0.0) -> float:
    """Success probability a verifier reports after misclassifying outcomes."""

    return p * (1 - false_negative) + (1 - p) * false_positive


def density_ratio(p: float, false_negative: float, false_positive: float = 0.0) -> float:
    """Reported signal density as a fraction of the true density.

    Signal density is proportional to p (1 - p), so this ratio says how a
    verifier moves the measured environment before any rollout is spent.
    """

    reported = shifted_probability(p, false_negative, false_positive)
    return reported * (1 - reported) / (p * (1 - p))


def density_slope(p: float) -> float:
    """First-order change in the density ratio per unit false-negative rate.

    Exact derivative at zero false-negative and zero false-positive rate.
    """

    return (2 * p - 1) / (1 - p)


def attenuation_rows() -> list[dict[str, Any]]:
    rows = []
    for false_negative in (0.10, 0.38, 0.50):
        for p in (0.035, 0.10, 0.25, 0.50, 0.75):
            ratio = density_ratio(p, false_negative)
            rows.append(
                {
                    "false_negative": false_negative,
                    "p": p,
                    "reported_p": round(shifted_probability(p, false_negative), 4),
                    "density_ratio": round(ratio, 4),
                    "loss": round(1 - ratio, 4),
                    "first_order_loss": round(-false_negative * density_slope(p), 4),
                }
            )
    return rows


def design_rows(p: float) -> list[dict[str, Any]]:
    rows = []
    for group_size in (2, 4, 8, 12, 16, 20, 24, 32):
        mean, variance = group_moments(p, group_size)
        rows.append(
            {
                "group_size": group_size,
                "tasks_per_budget": round(BUDGET_ROLLOUTS / group_size, 2),
                "expected_mass": round(mean, 4),
                "closed_form_mass": round(2 * group_size * p * (1 - p), 4),
                "variance": round(variance, 4),
                "mass_variance_over_group": round(variance / group_size, 6),
                "coverage": round(coverage(p, group_size), 3),
                "relative_se": round(pooled_relative_se(p, group_size), 3),
            }
        )
    return rows


def regimes() -> list[dict[str, Any]]:
    """The same table read at four success densities, to show where G pays."""

    result = []
    for p in (0.035, 0.10, 0.25, 0.50):
        rows = design_rows(p)
        best = min(rows, key=lambda row: row["relative_se"])
        result.append(
            {
                "p": p,
                "relative_se_at_g2": rows[0]["relative_se"],
                "relative_se_at_g8": rows[2]["relative_se"],
                "relative_se_best": best["relative_se"],
                "best_group_size": best["group_size"],
                "coverage_at_g2": rows[0]["coverage"],
                "coverage_at_g8": rows[2]["coverage"],
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", type=Path, help="write the design record to this path")
    arguments = parser.parse_args()

    p = PUBLISHED["success_rollouts"] / (PUBLISHED["tasks"] * PUBLISHED["rollouts_per_task"])
    print(f"pooled success probability p = {p:.4f}")
    print(f"tokens per rollout = {TOKENS_PER_ROLLOUT:.0f}")
    print()

    rebuilt = published_check()
    print("published record rebuilt from its own k vector")
    print(f"  advantage mass {rebuilt['advantage_mass']} against {PUBLISHED['advantage_mass']}")
    print(f"  H {rebuilt['H']} against {PUBLISHED['H']}")
    print(f"  coverage {rebuilt['coverage']:.2f} against {PUBLISHED['coverage']:.2f}")
    if rebuilt["H"] != PUBLISHED["H"] or rebuilt["advantage_mass"] != PUBLISHED["advantage_mass"]:
        raise SystemExit("published record does not rebuild from its own k vector")
    print()

    print("identity check: expected mass equals 2 G p (1 - p)")
    rows = design_rows(p)
    for row in rows:
        if abs(row["expected_mass"] - row["closed_form_mass"]) > 1e-9:
            raise SystemExit(f"closed form fails at G={row['group_size']}")
    print("  holds for every G in the table")
    print()

    print(f"design table at p = {p:.4f}, budget {BUDGET_ROLLOUTS} rollouts")
    print(" G  tasks  expected  variance  var/G  coverage  rel_se")
    for row in rows:
        print(
            f"{row['group_size']:2d}  {row['tasks_per_budget']:6.2f}  "
            f"{row['expected_mass']:8.4f}  {row['variance']:8.4f}  "
            f"{row['mass_variance_over_group']:7.4f}  {row['coverage']:8.3f}  "
            f"{row['relative_se']:6.3f}"
        )
    print()

    print("relative standard error across success densities")
    print("   p   rel_se(G=2)  rel_se(G=8)  best  G*  coverage(G=8)")
    for entry in regimes():
        print(
            f"{entry['p']:5.3f}  {entry['relative_se_at_g2']:11.3f}  "
            f"{entry['relative_se_at_g8']:11.3f}  {entry['relative_se_best']:5.3f}  "
            f"{entry['best_group_size']:3d}  {entry['coverage_at_g8']:13.3f}"
        )
    print()

    print("coverage targets at the pooled success probability")
    targets = {}
    for target in (0.25, 0.50, 0.75):
        group_size = coverage_group_size(p, target)
        targets[str(target)] = {
            "group_size": group_size,
            "coverage": round(coverage(p, group_size), 3),
            "tasks_per_budget": round(BUDGET_ROLLOUTS / group_size, 2),
        }
        print(
            f"  target {target:.2f}: G = {group_size}, "
            f"coverage {targets[str(target)]['coverage']:.3f}, "
            f"{targets[str(target)]['tasks_per_budget']:.1f} tasks at the same budget"
        )
    print()

    print("self test against seeded simulation (relative tolerance 0.02)")
    simulation = []
    for group_size in (4, 8, 16):
        exact_mean, exact_variance = group_moments(p, group_size)
        mean, variance = monte_carlo_moments(p, group_size)
        relative = abs(variance - exact_variance) / exact_variance
        simulation.append({"group_size": group_size, "relative_error": round(relative, 5)})
        print(
            f"  G={group_size:2d} exact {exact_variance:.4f} simulated {variance:.4f} "
            f"relative {relative:.4f}"
        )
        if relative > 0.02 or abs(mean - exact_mean) > 0.02 * exact_mean:
            raise SystemExit("simulation disagrees with the exact moments")
    print()

    print("verifier attenuation: reported signal density as a fraction of the true one")
    print("  fn   p      p_reported  ratio   loss    first_order")
    attenuation = attenuation_rows()
    for row in attenuation:
        print(
            f" {row['false_negative']:4.2f}  {row['p']:5.3f}  {row['reported_p']:10.4f}  "
            f"{row['density_ratio']:6.4f}  {row['loss']:6.3f}  {row['first_order_loss']:10.4f}"
        )
    for row in attenuation:
        close = row["false_negative"] <= 0.10 and row["p"] <= 0.25
        if close and abs(row["loss"] - row["first_order_loss"]) > 0.01:
            raise SystemExit("first-order attenuation disagrees with the exact ratio")
    print()

    record = {
        "schema_version": "anarres.signal-budget-theory.v1",
        "pooled_success_probability": round(p, 6),
        "budget_rollouts": BUDGET_ROLLOUTS,
        "tokens_per_rollout": round(TOKENS_PER_ROLLOUT, 1),
        "published_rebuild": rebuilt,
        "design_rows": rows,
        "regimes": regimes(),
        "coverage_targets": targets,
        "simulation_self_test": simulation,
        "verifier_attenuation": attenuation,
        "identities": {
            "expected_mass": "2 * G * p * (1 - p)",
            "mass_variance_over_group": "Var(A) / G, the budget-normalized variance",
            "coverage": "1 - p^G - (1 - p)^G",
        },
    }
    if arguments.write:
        arguments.write.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        digest = hashlib.sha256(arguments.write.read_bytes()).hexdigest()
        print(f"wrote {arguments.write} sha256:{digest}")
    else:
        print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
