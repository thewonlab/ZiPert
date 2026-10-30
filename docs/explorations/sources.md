# Sources and scope

*First public draft · October 30, 2026*

These pages summarize existing results. The accompanying assets are copies of already generated figures; no new model fitting was performed to prepare the posts. Full portable reproduction scripts and exact environment bundles are not included in this first draft.

## Figure S5

Source: the submitted biological study's finalized Figure S5 panel exports and summary CSVs, verified October 30, 2026.

- `SupplementaryFigure5E_mZINB_NB_discovery_counts.csv`: 60 / 5 / 2 emVar–gene discovery groups.
- `SupplementaryFigure5F_mZINB_NB_coefficient_audit_summary.csv`: common eligible tests, correlations and direction agreement.
- `SupplementaryFigure5G_mZINB_NB_hit_sparsity.png`: zero-fraction distributions among reported hits.
- `SupplementaryFigure5J_positive_control_FDR_plot_data.csv`: FOXP2, CALB2 and MEF2C adjusted values.
- `SupplementaryFigure5H_mZINB_NB_full_loglikelihood_audit_summary.csv`: likelihood comparison counts and units.

The historical analysis predates current development changes. We have not equated its fitter or QC settings with the current repository version. No clinical or participant-level records are included here.

## SPARSE01

Run date: October 29, 2026. Source summaries: `real_comparison.tsv`, `real_summary.json`, `real_diagnostics.png`, and the dated SPARSE01 experiment note. The source run retained the exact selected gene sets, model states, code snapshot, input hashes and configuration.

This pilot used the Microglia author's RNA counts rather than the independently deposited lane matrices, which differed from the author's aggregated counts. The selection seed was 2026102902. It used a development fitter snapshot and should not be treated as a benchmark of the unchanged public release.

## OPT02

Run summarized October 29, 2026. Source: the dated OPT02 results note and 120 retained experiment results. The experiment used selected generated cases, three seeds, and five strategies. Its custom fitting workflow and compute budgets differ across strategies; it is not a public-release speed comparison.

## Reading the status labels

**Draft** means the cited results exist, but the presentation is an evolving research note. **Outline / Planned** means the section does not yet contain a completed report. Completed simulation aggregation is distinguished from reviewed interpretation.
