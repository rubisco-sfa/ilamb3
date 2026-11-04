# Architecture decision records

These records state the decisions behind the scoring design of `ilamb3`,
and the reasoning for them is in [the design doc](../design/probabilistic-framework.md).

| Number | Title | Status |
|---|---|---|
| 0001 | [Probability theory as the design rule](0001-probability-theory-as-the-design-rule.md) | Accepted |
| 0002 | [Aggregate errors before scoring within an analysis](0002-aggregate-then-score.md) | Accepted |
| 0003 | [Save the statistics with each score map, and refuse to aggregate local scores](0003-save-statistics-with-score-maps.md) | Accepted |
| 0004 | [The Overall Score and the roll-up average normalized MSEs, then score once](0004-aggregate-then-score-across-analyses-and-datasets.md) | Accepted |
| 0005 | [Only orthogonal components of the MSE enter the total](0005-components-and-diagnostics.md) | Accepted rule; replacements for the canonical metrics proposed |
| 0006 | [Score a component on its error per unit volume](0006-volume-adjusted-component-scores.md) | Proposed |
| 0007 | [Define the ilamb3 score, distinct from a scoring function or a scoring rule](0007-definition-of-score.md) | Accepted |
| 0008 | [Use the score function exp(-alpha NMSE) with alpha = ln 2](0008-exponential-score-function.md) | Proposed |
