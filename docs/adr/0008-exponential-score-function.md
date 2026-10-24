---
status: proposed
---

# Use the score function exp(-alpha NMSE) with alpha = ln 2

ilamb3 scores an aggregated error with $e^{-E}$,
where $E = \sqrt{\mathrm{NMSE}}$.
We propose to score it with $s(\mathrm{NMSE}) = e^{-\alpha\,\mathrm{NMSE}}$ and $\alpha = \ln 2$,
so that the benchmark scores 0.5.
This score is a Gaussian likelihood ratio,
and it is the only continuous score
under which the scores of independent components multiply to the total score.

## Context

The `Collier2018` method in `ilamb3/analysis/scoring.py` maps the relative error $E$ to a score with $e^{-E}$ (`_score_function`, called by `score_nmse` and `score`).
This is Eq. 9 of Collier et al. (2018), $s = e^{-\alpha \varepsilon}$,
with $\alpha = 1$; they call the choice of the exponential arbitrary.
The `RegionalQuantiles` and `Linear` methods use $\max(0, 1 - E)$.
Under aggregate-then-score ([ADR 0002](0002-aggregate-then-score.md), [ADR 0004](0004-aggregate-then-score-across-analyses-and-datasets.md)) the score function is applied once,
to an aggregated NMSE,
so every strictly decreasing score function gives the same ranking of models.
The score function still decides what the numbers mean
and how the component scores on the scorecard relate to the total.
The argument is in [the score function](../design/probabilistic-framework.md#the-score-function).

## Decision

Proposed, not implemented:

- Score an NMSE with $s(\mathrm{NMSE}) = e^{-\alpha\,\mathrm{NMSE}}$,
  with $\alpha = \ln 2$, as the default for component scores, the Overall Score,
  and the roll-up.
- Keep the `Collier2018` method, $e^{-E}$ with $\alpha = 1$,
  for comparison with ILAMB v2.

The decision covers scores of an NMSE.
Diagnostics scored on a mean absolute error (`power=1` in `scoring.statistics`) are outside it.

## Justification

**Product rule.** The NMSE adds over orthogonal components ([ADR 0005](0005-components-and-diagnostics.md)).
The only continuous score with $s(0) = 1$ that satisfies $s(x + y) = s(x)\,s(y)$ is $s(x) = e^{-\alpha x}$ (Cauchy's exponential functional equation),
so $s(\sum_k \mathrm{NMSE}_k) = \prod_k s(\mathrm{NMSE}_k)$.
The current $e^{-\sqrt{x}}$ fails this rule: with $x = 0.7$ and $y = 1.9$,
$s(x + y) = 0.199$ and $s(x)\,s(y) = 0.109$ (design doc, check AB).
Hodson et al. (2021, Sec. 3.2, Eq. 15) score the NMSE with $e^{-\alpha\,\mathrm{NMSE}}$ and note that the product of the scores of orthogonal components equals the total score.

**Likelihood.** Suppose the errors are independent and Gaussian,
with variance $c$ times the reference's variance.
Then the likelihood ratio of the model to a perfect model is $e^{-\alpha\,\mathrm{NMSE}}$,
with $\alpha = n / (2c)$ for $n$ independent data (check AC).
Orthogonal components of such errors are independent,
so the likelihood factorizes over components.
The Gaussian is the distribution of maximum entropy
for errors summarized by their mean square (Jaynes, 2003, Chs. 7 and 11).
When no likelihood is trusted,
Bissiri et al. (2016) derive $e^{-w \cdot \mathrm{loss}}$ as the coherent update of beliefs with a loss in place of a likelihood;
the Gaussian likelihood is their case with squared error as the loss.
These derivations are in [the exponential of the NMSE](../design/probabilistic-framework.md#the-exponential-of-the-nmse).

**$\alpha$ is a convention** ([ADR 0001](0001-probability-theory-as-the-design-rule.md)).
$\alpha = n / (2c)$ depends on an error variance that is not known in practice,
and Bissiri et al. (2016) also leave the weight on the loss to be calibrated.
We fix $\alpha$ by the score of the benchmark, which has $\mathrm{NMSE} = 1$.
With $\alpha = \ln 2$ the benchmark scores 0.5,
a likelihood half that of a perfect model;
with $\alpha = 1$ it scores $e^{-1} = 0.368$.
$\alpha$ changes the displayed values and leaves every ranking unchanged.

## Considered options

- **Keep $e^{-E}$.** It fails the product rule,
  and the total has no closed form in the component scores.
- **$1/(1 + E)$ or $1/(1 + \mathrm{NMSE})$.** Neither is exponential in the NMSE,
  so neither satisfies the product rule.
  $1/(1 + \mathrm{NMSE})$ is the average of $e^{-\alpha\,\mathrm{NMSE}}$ over $\alpha \sim \mathrm{Exp}(1)$,
  the normalized Nash–Sutcliffe efficiency;
  its total is a harmonic mean of the component scores (check AD).
- **$\max(0, 1 - E)$** (`RegionalQuantiles`, `Linear`).
  It is not strictly decreasing:
  every comparison with $\mathrm{NMSE} \ge 1$ scores 0,
  so models worse than the benchmark tie.
- **A different $\alpha$.** Collier et al. (2018, Eq. 10) set $\alpha = -\ln(\hat s)/\hat\varepsilon$ to map a chosen error to a chosen score,
  and Hodson et al. (2021) use $\alpha = 3.14$ to spread scores across the unit interval.
  Any $\alpha$ gives the same rankings; $\ln 2$ ties the scale to the benchmark.

## Consequences

- Every score value changes; no ranking of models changes.
- Each component row's score is $e^{-\alpha\,\mathrm{NMSE}}$ of its `nmse` column.
  `scoring.add_overall_score` sums the `nmse` of the rows that share a normalizer label ([ADR 0004](0004-aggregate-then-score-across-analyses-and-datasets.md))
  and averages the sums across labels,
  so the Overall Score becomes the geometric mean, across normalizer labels,
  of the product of the component scores within each label.
  For a single normalizer it is the product of the component scores.
- The dashboard roll-up in `dataframe_to_cmec` (`ilamb3/meta.py`) scores the mean NMSE of a group,
  so the rolled-up score becomes the geometric mean
  of the scores of the datasets in the group.
- With volume-adjusted component scores ([ADR 0006](0006-volume-adjusted-component-scores.md)),
  the total is the mean-bias score times the volume-weighted geometric mean of the adjusted component scores ([score functions](../design/probabilistic-framework.md#score-functions));
  for the synthetic model of check O this is 0.1842 with $\alpha = \ln 2$ and 0.0871 with $\alpha = 1$ (checks AB and Q).
- Scores from ILAMB v2 and from the `Collier2018` method are not comparable in value with the new default.

## References

- Bissiri, P. G., Holmes, C. C., & Walker, S. G. (2016).
  A general framework for updating belief distributions.
  *Journal of the Royal Statistical Society Series B:
  Statistical Methodology*, 78(5), 1103–1130. https://doi.org/10.1111/rssb.12158
- Collier, N., Hoffman, F. M., Lawrence, D. M., Keppel-Aleks, G., Koven, C. D.,
  Riley, W. J., Mu, M., & Randerson, J. T. (2018).
  The International Land Model Benchmarking (ILAMB) system: Design, theory,
  and implementation. *Journal of Advances in Modeling Earth Systems*, 10(11),
  2731–2754. https://doi.org/10.1029/2018MS001354
- Hodson, T. O., Over, T. M., & Foks, S. S. (2021).
  Mean squared error,
  deconstructed. *Journal of Advances in Modeling Earth Systems*, 13,
  e2021MS002681. https://doi.org/10.1029/2021MS002681
- Jaynes, E. T. (2003). *Probability Theory: The Logic of Science*.
  Cambridge University Press. https://doi.org/10.1017/CBO9780511790423
