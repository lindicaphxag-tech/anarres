# Two levers on one scale

An environment contributes a success rate. A verifier contributes a false
negative rate and a false positive rate. Both move the same quantity, the
density that reaches the group advantage:

    p_effective = p (1 - a) + (1 - p) b

so the two levers can be priced against each other.

## False negatives are a discount on the environment

| base rate | clean | 20 percent | 38 percent |
| --- | --- | --- | --- |
| 0.035 | 1.000 | 0.806 | 0.629 |
| 0.100 | 1.000 | 0.818 | 0.646 |

Cutting false negatives from 38 to 20 percent at p = 0.035 recovers 1.28 times
the density, which is exactly what a 1.29 times higher base rate would give. A
verifier fix and an environment fix trade near one for one in this regime, so the
cheaper of the two is the one to buy.

## False positives are a subsidy to the wrong environment

| base rate | 2 percent | 10 percent |
| --- | --- | --- |
| 0.035 | 1.52 | 3.38 |
| 0.100 | 1.16 | 1.71 |

At p = 0.035 a ten percent false-positive rate reports 0.1142, which is 1.47
times the density of a clean environment that is genuinely 2.3 times better. The
contaminated arm wins the screen. Two percent is enough to manufacture a
phantom 1.5 times environment, and no amount of extra rollouts makes it
visible: more data narrows the interval around a biased number.

## The observation cannot attribute itself

A reported density of 0.0212 is consistent with p = 0.035 measured by a
38 percent false-negative verifier and with p = 0.0217 measured cleanly, a
38 percent gap in base rate. The two programmes are indistinguishable from the
number alone, and both are well inside what a 400-rollout panel resolves.
Attribution requires the verifier measured separately.

## What the audit costs

Labelled items, exact interval:

| rate | half width | items |
| --- | --- | --- |
| 0.38 | 0.03 | 1037 |
| 0.02 | 0.02 | 248 |

A thousand labelled items once is cheaper than a ten-candidate panel at 400
rollouts per arm, and without it the panel's ranks are uninterpretable. The audit
is the prerequisite, not the polish.

## Limits

- Flips are assumed independent and the rate homogeneous across tasks. The
  earlier grouped-flip cell moves outcomes differently, so a verifier that flips
  in blocks needs its own measurement.
- The audit needs known-answer items, and those come from fixtures with known
  outcomes rather than from the panel being screened.
- The exchange rates above are local to the stated base rates. At p = 0.25 the
  same false-negative rates cost more, because the density is steeper there.
