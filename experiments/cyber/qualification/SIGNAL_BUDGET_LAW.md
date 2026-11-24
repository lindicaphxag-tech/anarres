# What a fixed rollout budget buys

This note answers a budget question rather than a performance question: at a
fixed number of rollouts, which measurement do extra rollouts per task buy, and
which one do they give up? Three identities on the binomial group law settle it
without simulation, and the released 50-task sweep provides a check.

## Identities

Write p for a task's success probability, G for rollouts per task, k for
successes. The binary leave-one-out unnormalized advantage gives a group
A = 2 k (G - k) / (G - 1). Then:

1. E[A] = 2 G p (1 - p). Expected advantage mass carries no finite-G bias, so a
   group size is a design choice rather than a bias source.
2. Var(A) / G is the budget-normalized variance, because a fixed budget B buys
   B / G groups and the pooled estimate divides by B. At p = 0.035 this quantity
   runs from 0.1260 at G = 2 to 0.1171 at G = 32, a 7 percent change across a
   16-fold change in depth. At p = 0.5 it runs from 0.500 to 0.0161, a 31-fold
   change.
3. A group shows signal when 0 < k < G, which happens with probability
   1 - p^G - (1 - p)^G. At sparse p this probability is close to G p, at dense p
   it saturates near one.

Identity 2 is the design result. Pooling variance across a fixed budget,
Var(rho_hat) = Var(A) / (G B). In the sparse regime Var(A) grows almost
linearly in G, so the product is flat and the relative standard error of rho
hardly moves with depth. In the dense regime Var(A) saturates, the product falls
as 1 / G, and depth buys precision.

## Check against the released record

The 50-task sweep used G = 8, 400 rollouts, 14 successes, p = 0.035. From the
published k vector of twelve groups at 2.00 and one at 3.43, the normalized
concentration rebuilds to H = 0.2518 and the coverage to 13 / 50, both matching
the released summary. Identity 3 predicts coverage 0.248 at G = 8 and p = 0.035,
which is the same 26 percent.

| G | groups at 400 rollouts | coverage | relative standard error of rho |
| --- | --- | --- | --- |
| 2 | 200 | 0.068 | 0.263 |
| 8 | 50 | 0.248 | 0.254 |
| 20 | 20 | 0.510 | 0.254 |
| 32 | 12.5 | 0.680 | 0.253 |

The published interval [12.28, 33.85] is about plus or minus 50 percent around
22.42, and 1.96 times a relative standard error of 0.254 is 0.498. The released
interval is what exact theory predicts at this budget and this density, so no
reallocation of G narrows it. Halving that interval costs four times the
rollouts, 1,600 rather than 400.

## Design rule

- When p G is small, G is set by the coverage target and rho precision is bound
  to the budget. A 25 percent coverage target needs G = 9, 50 percent needs
  G = 20, 75 percent needs G = 39, each at the same 400 rollouts.
- When p is above roughly 0.25, depth buys precision: at p = 0.5 the same budget
  gives a 4.9 times tighter rho at G = 32 than at G = 2.
- A single sweep cannot serve both objectives at sparse p, because coverage costs
  depth in groups and precision costs groups in depth. Two panels, one wide and
  one deep, dominate one panel that splits the difference.

## Verifier error is a contraction, not a nuisance

Signal density is proportional to p (1 - p), so any verifier that misclassifies
outcomes applies an affine map to p and rescales the measured density by a
factor that depends only on p and the error rates. With a false-negative rate a
and a false-positive rate b, the reported probability is p (1 - a) + (1 - p) b
and the exact density ratio is reported (1 - reported) over p (1 - p).

At p = 0.035 the ratio is 0.629 for a = 0.38, so a verifier that misses 38
percent of successes hides 37 percent of the environment's signal. At p = 0.5
the first-order effect vanishes and the exact effect is a 14 percent loss. Above
p = 0.75 the same verifier raises the measured density by 33 percent, because
false negatives pull a mostly-succeeding task toward p = 0.5, where p (1 - p)
peaks. The map is a contraction toward one half, so verifier quality multiplies
signal rather than adding noise, and environments measured under different
verifier qualities cannot be ranked by rho without this correction.

## Falsifiable predictions

1. Two 400-rollout panels at G = 8 and G = 20 will report rho relative standard
   errors within 3 percent of each other at p = 0.035, while coverage moves from
   0.248 to 0.510.
2. Correcting a 38 percent false-negative rate at p = 0.035 raises the reported
   density by a factor of 1.59, and at p = 0.5 by a factor of 1.17.
3. A verifier with a 38 percent false-negative rate applied to a task at
   p = 0.75 will report a higher density than the clean verifier, by 33 percent.

## Reproduction

```bash
cd experiments/cyber/qualification
python3 signal_budget_theory.py
python3 signal_budget_theory.py --write design/signal_budget_v1.json
```

The script rebuilds H and coverage from the published k vector, checks identity
1 for every G in the table, checks the exact moments against a seeded
simulation, and checks the first-order attenuation against the exact ratio. Any
disagreement exits non-zero.
