# Contextual folded-count evidence (research candidate)

This extension of the folded-count diagnostic tests conditional taskwise folded-pair exchangeability, not equality of population mean reward variance or training utility. Let S be the orientation of the two folded counts. Given the past and the unordered current pair, the sign is fair on non-ties under the stated null. A context must be a function of the unordered pair or pre-outcome task metadata; using the orientation to choose it would invalidate the test.

Use three predeclared experts: one global sign mixture, one product of sign mixtures stratified by whether 4 times the folded gap exceeds group size, and one product stratified by group size exceeding 8. Each stratum uses the uniform-Beta sign wealth 2^(W+L)/[(W+L+1) choose(W+L,W)]. Given the context, the two possible next wealth ratios average exactly one. Only one stratum is updated per expert, so each expert is a nonnegative martingale; their equal-weight average is too. Ville's inequality bounds the probability of crossing 1/alpha at any time. Ties leave wealth unchanged. The maximum over experts alone would not have this guarantee. The test is two-sided and does not identify a directional mean effect. The exact calculation uses rational arithmetic; the synthetic benchmark uses floating point.

Related method: Saha and Ramdas, Testing exchangeability by pairwise betting (AISTATS 2024), https://proceedings.mlr.press/v238/saha24b.html. This is a scoped application and implementation of existing betting principles, not a claim of novel general statistical theory.

## Exploratory comparison

3,000 synthetic repetitions per scenario, 160 task pairs maximum, alpha=.05, seed 20261127. The same observations are fed to both tests. Null and single-direction alternatives use independent binomial arms with heterogeneous task sizes and probabilities. The gap-cancellation scenario is a deliberately constructed folded-pair stress case, not a binomial sampling model or empirical benchmark. Small and large gaps have opposite sign biases; marginal signs are fair. Contexts were fixed for this prototype, not selected by maximizing observed benchmark performance. Mean tasks include runs which did not reject at the horizon.

| Scenario | Global rejections | Contextual rejections | Global mean tasks | Contextual mean tasks |
|---|---:|---:|---:|---:|
| Null | 80/3000 | 95/3000 | 156.847 | 156.416 |
| Modest directional | 2997/3000 | 2998/3000 | 39.499 | 41.186 |
| Strong directional | 3000/3000 | 3000/3000 | 10.882 | 11.993 |
| Gap cancellation | 83/3000 | 3000/3000 | 156.603 | 23.389 |

The contextual test detects a failure mode of global sign aggregation, but pays a mixture penalty under simple directional alternatives. The results are not evidence of uniform superiority. No real task outcomes, GPU training, or new level-2 measurement were used. Seven local unit tests passed, including 441 rational martingale identities and 4096 adaptive-context paths. The benchmark checks 2560 exact-core prefix values against the vectorized implementation. Full repository CI was not run. This candidate is intentionally separate from the open PR until its scope is reviewed.

Run `python -m unittest -v` (standard library); `python benchmark.py` (NumPy) reproduces `results.json`.
