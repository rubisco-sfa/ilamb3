---
status: accepted
---

# Save the statistics with each score map, and refuse to aggregate local scores

A score map in an analysis output can be averaged over a region,
which computes a regional score in the score-then-average order
that [ADR 0002](0002-aggregate-then-score.md) removes.
`ilamb3` therefore saves every score map with the statistics and score method it was computed from,
flags it as a local score,
and refuses to aggregate a flagged array.
The order of operations becomes a rule the code checks,
and the saved statistics reproduce the regional score for any region
from the output files alone.

## Context

A score is a nonlinear function of an error,
so the mean of a score map depends on the score function
and can reorder models as its parameters change
([aggregating errors across components](../design/probabilistic-framework.md#aggregating-errors-across-components)).
A local score is still useful in a map,
where it shows in which cells a comparison's errors lie.
Once a score map is written to a netCDF file,
nothing in the file says it must not be averaged.

## Decision

- `scoring.add_score` in `ilamb3/analysis/scoring.py` stores the score map under its name,
  its statistics stacked along a `statistic` dimension in a variable named with the suffix `_statistics` (such as `biasscore_statistics`),
  and the method in that variable's attributes.
  For power 2 the statistics are `mse` and `mse_ref`;
  for power 1 they are `mae` and `mae_ref`.
- `scoring.score` flags any result that still varies over cells with `dataset.mark_local_score` in `ilamb3/dataset.py`,
  which sets the attribute `local_score = 1`.
  The attribute is saved in the output file.
- `dataset.check_aggregable` raises `ScoreAggregationError` (`ilamb3/exceptions.py`) when given a flagged array.
  The reductions that would average a score map call it:
  `dataset.integrate_space`, `dataset.integrate_time`,
  and the other reductions in `ilamb3/dataset.py` over space, time, or depth;
  the site branch of `base.integrate_or_mean` in `ilamb3/analysis/base.py`;
  and `scoring.aggregate`.
- The error message gives the fix:
  compute the regional score with `scoring.regional_score(ds, name, region)`,
  or aggregate the error map the score was computed from and score the result.
- `scoring.get_statistics(ds, name)` reads the statistics back,
  and `scoring.regional_score(ds, name, region)` aggregates them over the region
  and scores the result with the stored method.
- With mass weighting,
  an analysis multiplies the statistics by the cell weight with `scoring.weighted` before saving them.
  A score depends only on the ratio of the aggregated statistics,
  so the weighted statistics give the ratio of weighted means,
  and the saved statistics reproduce the mass-weighted score.

## Considered options

- **Do not save score maps.**
  This prevents the averaging,
  and also removes the map of where errors lie,
  which the Bias and RMSE analyses plot.
- **Save only the error map.**
  A regional score also needs the normalizer,
  and recomputing it requires the reference
  and the analysis's choice of normalizer,
  such as the variance in time or a regional quantile.
  The saved statistics carry the normalizer in `mse_ref` or `mae_ref`.
- **Document the rule without enforcing it.**
  A convention in documentation does not stop an analysis author or a user
  from averaging a score map,
  and the result has a plausible value in $[0, 1]$.
  The flag makes the mistake an error at the point it is made.

## Consequences

- Each score map adds one stacked statistics variable to the output file.
- `ilamb3/tests/test_analysis_conformance.py` requires every score map of every analysis to be flagged,
  and, for each map saved with statistics,
  checks that the global score recomputed from them
  matches the global score the analysis reports.
  `ilamb3/tests/test_score_aggregation.py` checks that each reduction refuses a flagged array,
  and that the error message says how to compute the regional score.
- A user who averages a score map with an `ilamb3` reduction
  gets an error that says how to compute the regional score.
- `ilamb3` sets `xr.set_options(keep_attrs=True)` in `ilamb3/__init__.py`,
  so an array derived from a score map, such as `1 - score`,
  keeps the flag and is refused too.
- Reductions called directly on xarray objects, such as `score.mean()`,
  are not checked.
- A regional score for a region that the analysis did not report
  can be computed from the output files without rerunning the analysis.

## References

- The user-facing description is in
  [local scores](../deep-dive/scoring.md#local-scores).
