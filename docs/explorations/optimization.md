# When pi moves but the likelihood barely changes

*Draft · October 30, 2026 · OPT02 development experiment*

## There is more than one kind of stability

A fit can return similar likelihoods with different zero-component estimates. Conversely, a difficult optimization problem can also change the likelihood-ratio statistic. We wanted to separate these possibilities rather than label every unstable pi estimate an optimizer failure.

## What we tried

OPT02 compared a baseline, bootstrap-derived initializations, a gamma-slope ridge path, alternating count/zero updates, and a hybrid candidate pool. Every final fit used the original data and an **unpenalized** joint likelihood. Bootstrap was used to find starting points, not to calibrate p-values.

There were **120 fits: eight selected generated datasets × three seeds × five strategies**. These were difficult examples selected from an earlier simulation, with two stable controls—not a representative random sample or new real-data fits. Gene names below identify the source profiles.

## Two contrasting examples

| Generated case, hybrid strategy | Range of delta-pi across seeds | Range of full log-likelihood | Range of LR |
|---|---:|---:|---:|
| STK3 / 3961 | 0.136 | 0.0081 | 0.0161 |
| OTUD6B / 973 | 0.0537 | 2.18 | 4.93 |

Here delta-pi is the standardized condition contrast, averaged over the same covariate distribution. A range across three seeds is not a confidence interval.

For STK3 / 3961, the estimated delta-pi ranged from about -0.088 to +0.048 while the full likelihood barely changed. The fit also had a dispersion clamp and a nonregular Hessian, so the sign of pi was not a defensible biological conclusion.

For OTUD6B / 973, the LR itself varied substantially. This is a counterexample to the tempting claim that the LRT is always safe whenever pi is unstable.

## Did combining strategies help?

The hybrid preserved earlier final candidates, so its likelihood could not be worse by construction. It improved the full likelihood beyond the best individual strategy in only **1/24 paired comparisons** (threshold 1e-6). Median total runtime was about **48.1 seconds**, versus **5.08 seconds** for the baseline. This was not an equal-compute-budget comparison.

The practical lesson is to retain good candidates and report parameter stability, likelihood stability, and inference diagnostics separately. We have not solved pi estimation in weakly informative data, and this experiment did not change package defaults.

[Source details](sources.md#opt02)
