# ZiPert explorations

Short notes on what we have observed while developing and using ZiPert. These are working posts, with figures and concrete examples rather than a finished methods paper.

**Updated: October 30, 2026 · First public draft**

Some notes describe a submitted biological study; others describe separate development experiments. They do not all use the same fitter, filters, or code version. A completed experiment is not automatically a validated statistical claim. These notes do not change the package defaults.

## Start here

- [Why we tried mZINB: a biological case study](case-study.md) — similar average-effect estimates, different discoveries, and positive controls.
- [Is a high zero fraction enough?](sparse-regimes.md) — a focused look at sparse genes with and without informative positive counts.
- [When pi moves but the likelihood barely changes](optimization.md) — what our optimizer experiments did and did not stabilize.

## Six questions we are working on

| Research axis | What belongs here | Status |
|---|---|---|
| 1. [Optimization stability](optimization.md) | Initializations, likelihood preservation, convergence and parameter reproducibility. | Draft observations |
| 2. [Identifiability and model choice](model-choice.md) | When detection models, mZINB or NB are useful, and when inference should be withheld. | Outline + sparse-data note |
| 3. [DEG benchmarking](deg-benchmarking.md) | Error rates, power, effect estimates and reproducibility against other DEG tools. | Case study + outline |
| 4. [Zero-structure inference](zero-inference.md) | Perturbation effects on the zero component beyond changes in marginal mean. | Planned |
| 5. [Sharing the zero model](shared-zero.md) | Fully shared, gene-specific intercept/shared effect, and independent zero models. | Planned |
| 6. [Biological interpretation](biology.md) | Testing connections to transcriptional ON/OFF behavior against technical explanations. | Planned |

[Sources and scope](sources.md) records where the figures and numerical summaries came from. Full portable reproduction bundles and broader DEG-tool benchmarks are still being assembled.
