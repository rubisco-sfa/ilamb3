---
status: accepted
---

# Probability theory as the design rule

ilamb3 combines errors over cells, times, components, analyses, and datasets,
and each way of combining them is a design choice.
We derive these choices, as far as practicable,
from the rules of probability theory and a stated error model,
and we label every choice that does not follow from them as a convention,
a modeling choice, or a value judgment.
A ranking or roll-up derived this way is reproducible,
its assumptions can be checked,
and every remaining choice is visible as a choice.

## Context

Model evaluation is inference: from incomplete and uncertain observations,
we judge how well a model represents a process,
and we choose among models on that judgment.
Cox (1946) showed that a rule for representing plausibility by real numbers
must follow the product and sum rules of probability theory,
up to a monotone change of scale,
if it agrees with deductive logic
and gives the same result by every valid route of reasoning.
Jaynes (2003, Chs. 1–2) developed this result
into probability theory as the logic of plausible reasoning.
In ilamb3,
the requirement that every route give the same result
is consistency in aggregation:
a total computed in stages equals the total computed at once.
The argument is in the [Abstract](../design/probabilistic-framework.md#abstract)
and [The design rule](../design/probabilistic-framework.md#the-design-rule)
of the design doc.

## Decision

As far as practicable,
ilamb3 is consistent with and based on the rules of probability theory,
and those rules are considered first in every design choice.
A choice that does not follow from them is allowed,
but it is stated as one of these kinds (the terms are defined in `CONTEXT.md`):

- **Derived choice.** The choice follows from the rules of probability theory
  and a stated error model.
- **Convention.** The choice changes only how results are displayed,
  never which conclusions follow.
- **Modeling choice.** The choice encodes prior information,
  such as the benchmark or the baseline.
- **Value judgment.** The choice expresses a scientific priority
  and enters as a utility (Jaynes, 2003, Chs. 13–14).

`AGENTS.md` states this rule for contributors,
together with its two working forms: aggregate-then-score,
and add like with like.

The main choices, from the design doc's table:

| Choice | Kind | ADR |
|---|---|---|
| Aggregate the squared error, then score | Derived | [ADR 0002](0002-aggregate-then-score.md), [ADR 0004](0004-aggregate-then-score-across-analyses-and-datasets.md) |
| Common units and normalizer for summed terms | Derived | [ADR 0004](0004-aggregate-then-score-across-analyses-and-datasets.md), [ADR 0005](0005-components-and-diagnostics.md) |
| Orthogonal components in the total, other metrics as diagnostics | Derived | [ADR 0005](0005-components-and-diagnostics.md) |
| Component scores on the error per unit volume | Derived | [ADR 0006](0006-volume-adjusted-component-scores.md) (proposed) |
| Score function $e^{-\alpha\,\mathrm{NMSE}}$ | Derived | [ADR 0008](0008-exponential-score-function.md) (proposed) |
| $\alpha = \ln 2$ | Convention | [ADR 0008](0008-exponential-score-function.md) (proposed) |
| Error interval $b \pm 2 s_e$ | Derived | None |
| Benchmark | Modeling choice | None |
| Baseline covariance $\rho$ | Modeling choice | [ADR 0006](0006-volume-adjusted-component-scores.md) (proposed) |
| Weights on datasets by reliability | Derived | None |
| Weights on variables, datasets, and regions by importance | Value judgment | None; [ADR 0004](0004-aggregate-then-score-across-analyses-and-datasets.md) keeps equal weights |

[ADR 0007](0007-definition-of-score.md)
defines the score that these choices produce.
A row marked None has no ADR yet; its basis is in the design doc.

## Justification

Under this rule,
a ranking or roll-up follows from an error model
and the rules of probability theory,
so a reader can rerun it and check its assumptions.
For example, the squared error adds over the data as a log-likelihood does,
so it is aggregated before a monotone score function is applied,
and the ranking of models does not depend on that function
([ADR 0002](0002-aggregate-then-score.md)).
The choices that probability theory leaves open are labeled by kind,
so a reader can tell which conclusions follow from the evidence
and which follow from priorities.
The numerical claims in the design doc come from the scripts `docs/design/experiments/verify_*.py`,
listed in its [Verification](../design/probabilistic-framework.md#verification) section,
and are rerun when a claim or the scoring code it describes changes.

## Considered options

### Separately designed metrics, combined by averages of scores

ILAMB v2 scores each metric on its own
and averages the scores with weights:
the RMSE score counts twice in the Overall Score,
and datasets and variables are weighted through the configuration
(Collier et al., 2018, Eq. 31 and Sec. 2.4).
Until the change recorded in [ADR 0004](0004-aggregate-then-score-across-analyses-and-datasets.md),
ilamb3 took the unweighted mean of the score rows for the Overall Score
and of the Overall Scores in the roll-up.
An average of scores depends on the score function,
so two monotone score functions can rank the same models differently,
and the weights have no stated basis.
We rejected it.

### Full Bayesian model comparison

Jaynes (2003, Ch. 20) compares models by their probabilities given the data.
With a stated Gaussian error model,
the proposed score $e^{-\alpha\,\mathrm{NMSE}}$ is the likelihood ratio in that comparison,
with $\alpha = n / (2c)$ for $n$ independent data and an error variance $c$ times the reference's variance
([the exponential of the NMSE](../design/probabilistic-framework.md#the-exponential-of-the-nmse)).
That variance is not known in practice,
so $\alpha$ is fixed by convention ([ADR 0008](0008-exponential-score-function.md)),
and the score is then a tempered likelihood (Bissiri et al., 2016).
ilamb3 therefore follows the rules of probability theory
in how it combines errors,
and it does not compute posterior probabilities of models.

## Consequences

- Each ADR states the kind of each choice it makes.
- A new metric states its error model and enters the total as a component,
  or it is labeled a diagnostic and kept out of the total and every roll-up
  ([ADR 0005](0005-components-and-diagnostics.md)).
- Changes to how errors are normalized, decomposed, aggregated,
  or scored start from [the design doc](../design/probabilistic-framework.md),
  as `AGENTS.md` requires.

## References

- Bissiri, P. G., Holmes, C. C., & Walker, S. G. (2016).
  A general framework for updating belief distributions.
  *Journal of the Royal Statistical Society Series B:
  Statistical Methodology*, 78(5), 1103–1130.
  https://doi.org/10.1111/rssb.12158
- Collier, N., Hoffman, F. M., Lawrence, D. M., Keppel-Aleks, G., Koven, C. D.,
  Riley, W. J., Mu, M., & Randerson, J. T. (2018).
  The International Land Model Benchmarking (ILAMB) system: Design, theory,
  and implementation. *Journal of Advances in Modeling Earth Systems*, 10(11),
  2731–2754. https://doi.org/10.1029/2018MS001354
- Cox, R. T. (1946).
  Probability,
  frequency and reasonable expectation. *American Journal of Physics*, 14(1),
  1–13. https://doi.org/10.1119/1.1990764
- Jaynes, E. T. (2003). *Probability Theory: The Logic of Science*.
  Cambridge University Press. https://doi.org/10.1017/CBO9780511790423
