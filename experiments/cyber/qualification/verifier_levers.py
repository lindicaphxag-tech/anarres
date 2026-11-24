"""Two levers on the same scale: the base rate and the verifier.

An environment contributes a success rate p. A verifier contributes a false
negative rate a and a false positive rate b. The rate that actually reaches the
group advantage is

    p_effective = p (1 - a) + (1 - p) b

and signal density is p_effective (1 - p_effective). Both levers move the same
quantity, so they can be priced against each other: what a verifier improvement
is worth in base-rate terms, and what a contaminated verifier does to the ranking.

The third result is the uncomfortable one. A density observed under an unknown
verifier cannot be attributed, because a better environment measured badly and a
worse environment measured well produce the same number. Attribution needs the
verifier measured, and the measurement has a price in labelled items, not
rollouts.
"""

from __future__ import annotations

import math

from scipy.stats import beta

BASELINE = 0.035


def effective_rate(p: float, false_negative: float = 0.0, false_positive: float = 0.0) -> float:
    return p * (1.0 - false_negative) + (1.0 - p) * false_positive


def density(p: float) -> float:
    return p * (1.0 - p)


def observed_density(p: float, false_negative: float = 0.0, false_positive: float = 0.0) -> float:
    return density(effective_rate(p, false_negative, false_positive))


def rate_for_density(value: float) -> float:
    """Lower of the two base rates that produce a density."""

    discriminant = 1.0 - 4.0 * value
    if discriminant <= 0:
        return 0.5
    return (1.0 - math.sqrt(discriminant)) / 2.0


def lever_price(p: float, before: float, after: float) -> dict[str, float]:
    """What cutting false negatives from before to after is worth in base rate."""

    low = rate_for_density(observed_density(p, before))
    high = rate_for_density(observed_density(p, after))
    return {
        "recovered_density": observed_density(p, after) / observed_density(p, before),
        "equivalent_rate_before": low,
        "equivalent_rate_after": high,
        "base_rate_equivalent": high / low,
    }


def contamination(p: float, false_positive: float, rival: float = 0.0849) -> dict[str, float]:
    """How a false-positive rate changes the ranking of a weak and a strong arm."""

    dirty = observed_density(p, false_positive=false_positive)
    clean_rival = density(rival)
    return {
        "dirty_density": dirty,
        "clean_rival": clean_rival,
        "dirty_over_rival": dirty / clean_rival,
        "misranks": 1.0 if dirty > clean_rival else 0.0,
    }


def attribution_family(observed: float, false_negative: float) -> float:
    """Base rate that would produce the observation if the verifier is this bad."""

    return observed / (1.0 - false_negative)


def ambiguity() -> dict[str, float]:
    """The pair that cannot be told apart from density alone."""

    good_with_bad_verifier = observed_density(0.035, false_negative=0.38)
    return {
        "observed": good_with_bad_verifier,
        "rate_bad_verifier": attributed(good_with_bad_verifier, 0.38),
        "rate_clean_verifier": attributed(good_with_bad_verifier, 0.0),
        "gap": abs(attributed(good_with_bad_verifier, 0.0) - 0.035) / 0.035,
    }


def attributed(observed_density_value: float, false_negative: float) -> float:
    return rate_for_density(observed_density_value) / (1.0 - false_negative)


def audit_items(rate: float, half_width: float) -> int:
    """Labelled items needed to pin a rate within a half width, exact interval."""

    low, high = 2, 200000
    while low < high:
        middle = (low + high) // 2
        successes = round(rate * middle)
        lower = beta.ppf(0.025, successes, middle - successes + 1) if successes > 0 else 0.0
        upper = beta.ppf(0.975, successes + 1, middle - successes) if successes < middle else 1.0
        if upper - lower <= 2.0 * half_width:
            high = middle
        else:
            low = middle + 1
    return low


def main() -> None:
    print("p     alpha  beta   p_eff   density ratio")
    for p in (0.035, 0.10):
        for alpha, beta_rate in ((0.0, 0.0), (0.38, 0.0), (0.20, 0.0), (0.0, 0.02), (0.0, 0.10)):
            print(
                "%.3f  %.2f  %.2f   %.4f  %.3f"
                % (
                    p,
                    alpha,
                    beta_rate,
                    effective_rate(p, alpha, beta_rate),
                    observed_density(p, alpha, beta_rate) / density(p),
                )
            )
    price = lever_price(BASELINE, 0.38, 0.20)
    print()
    print(
        "cutting false negatives 0.38 to 0.20 at 0.035 recovers %.3f of the density, "
        "which equals a %.2f times base rate"
        % (price["recovered_density"], price["base_rate_equivalent"])
    )
    dirty = contamination(BASELINE, 0.10)
    print(
        "a 0.10 false-positive rate lifts 0.035 to %.4f, which is %.2f times a clean "
        "2.3 times leader and misranks: %d"
        % (dirty["dirty_density"], dirty["dirty_over_rival"], dirty["misranks"])
    )
    ambiguity_row = ambiguity()
    print()
    print(
        "the same observed density 0.0212 comes from p=0.035 with 38 percent false "
        "negatives or p=%.4f with a clean verifier, a %.0f percent gap"
        % (ambiguity_row["rate_clean_verifier"], ambiguity_row["gap"] * 100)
    )
    print(
        "audit items: 0.38 to within 0.03 -> %d, 0.02 to within 0.02 -> %d"
        % (audit_items(0.38, 0.03), audit_items(0.02, 0.02))
    )
    if observed_density(BASELINE, 0.38, 0.0) >= density(BASELINE):
        raise SystemExit("false negatives did not reduce the density")
    if contamination(BASELINE, 0.10)["misranks"] == 0.0:
        raise SystemExit("a contaminated verifier did not misrank the panel")
    if ambiguity_row["gap"] < 0.3:
        raise SystemExit("the attribution ambiguity is smaller than claimed")
    if audit_items(0.38, 0.03) > 4000:
        raise SystemExit("the audit costs more than the panel it protects")


if __name__ == "__main__":
    main()
