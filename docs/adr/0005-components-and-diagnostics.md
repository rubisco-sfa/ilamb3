---
status: accepted
---

# Only orthogonal components of the MSE enter the total

A sum of errors double counts any error that two of its terms share,
so a metric enters the Overall Score, or any roll-up of it,
only if it is one term of a single orthogonal decomposition of the MSE,
counted once.
Every other metric is a diagnostic: computed and shown on the scorecard,
but excluded from the total and from every roll-up.
The rule is accepted and enforced on this branch through the `nmse` column of the results table;
replacing the canonical metrics with their orthogonal counterparts is proposed.

## Context

The Overall Score adds normalized squared errors instead of averaging scores,
as [ADR 0002](0002-aggregate-then-score.md)
and [ADR 0004](0004-aggregate-then-score-across-analyses-and-datasets.md) decide.
A sum of errors requires terms in the same units, divided by one normalizer,
that do not overlap;
see [Aggregating errors across components](../design/probabilistic-framework.md#aggregating-errors-across-components).
On `main`, `add_overall_score` averaged every score row with equal weight.
The Spatial Distribution and Seasonal Cycle scores therefore counted again
errors that the Bias and RMSE scores already contain.

## Decision

A metric contributes to the total only if it is one term, counted once,
of a single orthogonal decomposition of the MSE;
see [Rule](../design/probabilistic-framework.md#rule).
Two kinds of metric fail the rule: metrics that are not components of the MSE,
such as the Taylor score, the phase shift, the Hellinger distance,
and the relationship error; and components, or coarser versions of components,
that are already in the total.

On this branch, a score row is a component if it carries an `nmse`,
the aggregated NMSE behind the score, and a `normalizer` label.
[ADR 0004](0004-aggregate-then-score-across-analyses-and-datasets.md) describes how `scoring.add_overall_score` combines them.
Rows without an `nmse` are diagnostics;
`add_overall_score` and the dashboard roll-up in `dataframe_to_cmec` (`ilamb3/meta.py`) leave them out.

| Score | On this branch | Orthogonal counterpart (proposed) |
|---|---|---|
| Bias Score | Component | Mean bias + pattern |
| RMSE Score | Component | Seasonal + anomaly |
| Hydro, annual mean | Component | The bias term |
| nbp Trajectory Score | Component | — |
| Time Series Bias Score | Component | The series' mean bias |
| Spatial Distribution | Diagnostic | Pattern amplitude, pattern correlation |
| Seasonal Cycle | Diagnostic | Seasonal timing (with seasonal amplitude) |
| Dispersion | Diagnostic | Seasonal and anomaly amplitude parts |
| Hydro, seasonal | Diagnostic | Season-mean part of the seasonal term (an alternative split) |
| Time Series Taylor Score | Diagnostic | The series' amplitude and correlation parts |
| Area Missed and Excess Scores | Diagnostic | Missed area and excess area, each over the reference area |
| nbp Difference Score | Diagnostic | The evaluation year's share of the Trajectory error |
| Runoff Sensitivity scores | Diagnostic | Each sensitivity's error, normalized in its own units |
| Relationship | Diagnostic | None |

The counterparts come from [the design doc's table](../design/probabilistic-framework.md#canonical-metrics-and-their-orthogonal-counterparts).
The Runoff Sensitivity scores have a normalizer of 1,
so their errors are dimensional.

The following are proposed and not implemented:

- the seven-term default decomposition,
  in place of Spatial Distribution and Seasonal Cycle in the total: mean bias,
  pattern amplitude and correlation, seasonal amplitude and timing,
  and anomaly amplitude and correlation;
  see [Default decomposition](../design/probabilistic-framework.md#default-decomposition);
- moving the derived variables out of the roll-up, such as NEE, net radiation,
  albedo, and evaporative fraction;
  see [Roll-up across variables](../design/probabilistic-framework.md#roll-up-across-variables);
- a separate `"diagnostic"` row type,
  and a muted scorecard style for diagnostics;
  see [Scorecard](../design/probabilistic-framework.md#scorecard),
  and [Implementation](../design/probabilistic-framework.md#implementation).

## Justification

Take the projections of the error onto nested subspaces: the constants,
the fields constant in time, the fields of cell and calendar month,
and all fields.
Their increments are mutually orthogonal,
so their squared norms sum to the MSE with no cross terms;
see [Four orthogonal terms](../design/probabilistic-framework.md#four-orthogonal-terms).
Under a Gaussian error model, orthogonal components are independent,
as [the design rule](../design/probabilistic-framework.md#the-design-rule) states,
so their sum counts no error twice.
The squared bias and the squared centered RMSE are such components,
as in Murphy (1988, Eqs. 8–9); with one normalizer and no uncertainty discount,
their NMSEs sum to the dataset's NMSE.

A metric that is not a component does one of two things:
it measures in other units, or it overlaps a term already in the total.

- The Seasonal Cycle phase shift is in months.
  It is not a function of the seasonal term:
  of two models with equal seasonal error,
  one scores 1.00 and the other 0.67 (check G).
- The Spatial Distribution score is a function of the Taylor statistics,
  from Taylor (2001), that is not a function of the pattern term.
  It ranks first a model whose pattern error is 60% larger (check D).
- The Hellinger distance of Dispersion compares probability distributions;
  it responds to differences in mean and spread,
  which Bias and RMSE contain (fix 8).
- Each Hydro seasonal bias contains the annual bias,
  which the annual score already counts;
  their spread about it is a coarser version of the seasonal term
  (fix 7; check K).
- The nbp Difference Score measures one year of the error,
  and the Trajectory Score pools that year (fix 13).
- The relationship error responds to a bias in the dependent variable (fix 12).
  The driver-conditional split of the MSE is orthogonal,
  but it cuts across the decomposition,
  and it does not measure a response to each source's own driver (check N).

Two splits of the same term are not orthogonal to each other in general,
so the total uses one split per term, and it reports the others as diagnostics.
For example, Gupta et al. (2009) split the pattern term into nonnegative parts,
which are not orthogonal (check B);
the distribution and sequence terms of Hodson et al. (2021, Sec. 3.5)
are additive, but not orthogonal.
The design doc's checks A–N verify these identities on synthetic data,
in `docs/design/experiments/verify_components.py`, `verify_fixes.py`,
and `verify_suite.py`.

## Considered options

- Keep every canonical metric in the total, with weights.
  ILAMB v2 did this,
  with the RMSE score counted twice (Collier et al., 2018, Eq. 31).
  Weights do not remove the overlap between scores,
  and they cannot make a phase shift in months addable to a mean square.
- Keep every canonical metric in the total, with equal weights.
  This is `main`.
  The overlap remains: with the default analyses,
  the Bias and RMSE errors are counted again,
  through Spatial Distribution and Seasonal Cycle.

## Consequences

- Until the orthogonal counterparts are implemented,
  the Overall Score does not reflect the Seasonal Cycle, Spatial Distribution,
  or Relationship scores.
  It reflects the errors they target only through the Bias and RMSE terms.
- Hydro's seasonal scores overlap its annual score, so they are diagnostics.
- If the Time Series analysis runs on a gridded dataset with Bias and RMSE,
  its Bias Score overlaps the mean bias term,
  because the MSE of the regional-mean series is already in the total (check L).
  Its normalizer label differs,
  so `add_overall_score` averages it with the Bias and RMSE sum instead of adding it.
- The sum of the Bias and RMSE NMSEs equals the dataset's NMSE,
  but only without the uncertainty discount,
  which both analyses still apply (fix 5).

## References

- Collier, N., Hoffman, F. M., Lawrence, D. M., Keppel-Aleks, G., Koven, C. D.,
  Riley, W. J., Mu, M., & Randerson, J. T. (2018).
  The International Land Model Benchmarking (ILAMB) system:
  Design, theory, and implementation.
  *Journal of Advances in Modeling Earth Systems*, 10(11), 2731–2754.
  https://doi.org/10.1029/2018MS001354
- Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009).
  Decomposition of the mean squared error and NSE performance criteria:
  Implications for improving hydrological modelling.
  *Journal of Hydrology*, 377(1–2), 80–91.
  https://doi.org/10.1016/j.jhydrol.2009.08.003
- Hodson, T. O., Over, T. M., & Foks, S. S. (2021).
  Mean squared error, deconstructed.
  *Journal of Advances in Modeling Earth Systems*, 13, e2021MS002681.
  https://doi.org/10.1029/2021MS002681
- Murphy, A. H. (1988).
  Skill scores based on the mean square error
  and their relationships to the correlation coefficient.
  *Monthly Weather Review*, 116(12), 2417–2424.
  https://doi.org/10.1175/1520-0493(1988)116<2417:SSBOTM>2.0.CO;2
- Taylor, K. E. (2001).
  Summarizing multiple aspects of model performance in a single diagram.
  *Journal of Geophysical Research: Atmospheres*, 106, 7183–7192.
  https://doi.org/10.1029/2000JD900719
