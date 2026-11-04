---
status: accepted
---

# Define the ilamb3 score, distinct from a scoring function or a scoring rule

The words "score" and "scoring" have established meanings
in forecast verification that differ from ILAMB's.
ilamb3 keeps both words and defines them:
a score is a strictly decreasing transformation
of an aggregated normalized error, used for display and ranking,
and it is neither a scoring function in the sense of Gneiting (2011)
nor a scoring rule in the sense of Gneiting and Raftery (2007).

## Decision

An **ilamb3 score** is a number in $[0, 1]$, where 1 is perfect,
obtained by applying a strictly decreasing score function $s$
to an aggregated normalized error
of a comparison against a reference over a domain:
a region, a period, or a set of datasets.
For most analyses the error is the NMSE,
or its square root $E = \sqrt{\mathrm{NMSE}}$;
for Seasonal Cycle and Dispersion, whose scores are affine in their error,
it is the ratio of mean absolute errors, `mae / mae_ref`.

**Scoring** is the procedure:
compute the error, normalize it, aggregate it, and apply the score function once
([ADR 0002](0002-aggregate-then-score.md), [ADR 0004](0004-aggregate-then-score-across-analyses-and-datasets.md)).
In `ilamb3/analysis/scoring.py`, `scoring.statistics` computes the additive statistics,
`scoring.aggregate` aggregates them,
`scoring.nmse` normalizes them,
and `scoring.score_nmse` applies the score function.

A **local score** is the score function applied to one cell's statistics.
It is for display in a map and is never averaged into a regional score
([ADR 0003](0003-save-statistics-with-score-maps.md)).

The **score function** is the map $s$.
ilamb3 implements $e^{-E}$ (`Collier2018`)
and $\max(0, 1 - E)$ (`RegionalQuantiles` and `Linear`);
[ADR 0008](0008-exponential-score-function.md) proposes $e^{-\alpha\,\mathrm{NMSE}}$ with $\alpha = \ln 2$.
The clipped functions are strictly decreasing for $E < 1$
and tie every error with $E \ge 1$ at 0.

A score is a monotone summary.
Because the error is aggregated before $s$ is applied,
every strictly decreasing $s$ ranks models in the same order
([the score function](../design/probabilistic-framework.md#the-score-function)).

## Justification

Gneiting (2011) defines a scoring function $S(x, y)$
as the loss when the point forecast $x$ is issued and the observation $y$ materializes,
taken to be negatively oriented (smaller is better).
A scoring function is consistent for a functional
when the expected score is minimized by reporting that functional;
the squared error $S(x, y) = (x - y)^2$ is consistent for the mean.

Gneiting and Raftery (2007) define a scoring rule $S(P, x)$
as a function of a probabilistic forecast $P$
and the event or value $x$ that materializes.
They take scoring rules to be positively oriented rewards.
A scoring rule is proper
when a forecaster whose best judgment is $Q$
maximizes the expected score by quoting $P = Q$,
and strictly proper when that maximum is unique.

In these terms, ilamb3 uses the squared error as its scoring function.
The NMSE is the average of that scoring function over the domain,
divided by the benchmark's average.
The ilamb3 score is a strictly decreasing transformation of that ratio.
It evaluates a point forecast, the comparison, so it is not a scoring rule,
and it is a transformation of an average of losses,
so it is not a scoring function.

The MSE skill score $1 - \mathrm{NMSE}$ (Murphy, 1988)
is another summary of the same ratio, unbounded below,
and is the skill score of [CONTEXT.md](../../CONTEXT.md).
Gneiting and Raftery (2007, Sec. 2.3) note
that skill scores of this standardized form are generally improper,
even when the underlying scoring rule is proper.

"Score function" (ilamb3's map $s$)
and "scoring function" (Gneiting's loss $S$) differ by one word.
Docs say which is meant.

## Considered options

- **Rename the score**, for example to "rating" or "skill".
  This would break the API (`scoring.score`, `scoring.add_score`),
  the score names in the results CSV and the dashboard,
  and the usage of ILAMB since Collier et al. (2018).
  "Skill" also names $1 - \mathrm{NMSE}$, a different quantity.
- **Adopt Gneiting's term** and call the score a scoring function.
  Gneiting's scoring function is the loss for one forecast and one observation;
  the ilamb3 score is a summary of many losses
  after normalization and aggregation.

## Consequences

- [CONTEXT.md](../../CONTEXT.md), the user [glossary](../reference/glossary.md),
  and the [Scoring](../deep-dive/scoring.md) page carry this definition.
- Code names are unchanged.
- "Scoring function" and "scoring rule" appear in ilamb3 docs
  only in the senses of Gneiting (2011) and Gneiting and Raftery (2007).

## References

- Collier, N., Hoffman, F. M., Lawrence, D. M., Keppel-Aleks, G., Koven, C. D.,
  Riley, W. J., Mu, M., & Randerson, J. T. (2018).
  The International Land Model Benchmarking (ILAMB) system:
  Design, theory, and implementation.
  *Journal of Advances in Modeling Earth Systems*, 10(11), 2731–2754.
  https://doi.org/10.1029/2018MS001354
- Gneiting, T. (2011).
  Making and evaluating point forecasts.
  *Journal of the American Statistical Association*, 106(494), 746–762.
  https://doi.org/10.1198/jasa.2011.r10138
- Gneiting, T., & Raftery, A. E. (2007).
  Strictly proper scoring rules, prediction, and estimation.
  *Journal of the American Statistical Association*, 102(477), 359–378.
  https://doi.org/10.1198/016214506000001437
- Murphy, A. H. (1988).
  Skill scores based on the mean square error
  and their relationships to the correlation coefficient.
  *Monthly Weather Review*, 116(12), 2417–2424.
  [https://doi.org/10.1175/1520-0493(1988)116<2417:SSBOTM>2.0.CO;2](https://doi.org/10.1175/1520-0493(1988)116<2417:SSBOTM>2.0.CO;2)
