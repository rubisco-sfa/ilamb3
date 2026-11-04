# ilamb3

`ilamb3` evaluates land models by comparing their output with reference datasets and turning the errors into scores.
These are the terms for evaluation as inference under the rules of probability theory;
the reasoning behind them is in [docs/design/probabilistic-framework.md](docs/design/probabilistic-framework.md).

## Language

### Datasets and errors

**Reference**:
The dataset treated as the truth in an evaluation, usually observations.
_Avoid_: truth, target

**Comparison**:
The dataset evaluated against the reference, usually the output of a model.
_Avoid_: model output, simulation (for the dataset)

**Model**:
The system that produced a comparison; scores rank models.

**Analysis**:
One way of comparing a comparison with a reference, such as Bias, RMSE, Seasonal Cycle, or Relationship.
_Avoid_: metric

**Error**:
The difference between the comparison and the reference at each point, $e = f - x$.
_Avoid_: residual, misfit

**Benchmark**:
The no-skill comparison that defines the normalizer: the reference's own mean over the period, in each cell or over the region.
Its NMSE is 1.
_Avoid_: baseline, climatology

**Normalizer**:
The variance $D$ of the reference about the benchmark, which divides every squared error of a dataset.
_Avoid_: scale, denominator

**NMSE**:
The normalized mean squared error: the mean squared error divided by the normalizer.
_Avoid_: relative error (its square root)

### Scoring

**Score**:
A value from 0 to 1, where 1 is perfect, given by applying the score function once to a normalized error aggregated over a domain: a region, a period, or a set of datasets.
It summarizes losses for display and ranking; it is neither a scoring function (Gneiting, 2011) nor a scoring rule (Gneiting and Raftery, 2007).
_Avoid_: scoring function, scoring rule (for the score)

**Scoring**:
The procedure that produces a score: compute the error, normalize it, aggregate it, and apply the score function once.

**Scoring function**:
In the sense of Gneiting (2011) only: the loss of one point forecast against one observation, such as the squared error, which `ilamb3` averages into the NMSE.

**Score function**:
The strictly decreasing map $s$ from an error to a score, such as $e^{-\alpha\,\mathrm{NMSE}}$.
_Avoid_: score (for the map), scoring function (a loss), scoring rule (a term for probabilistic forecasts)

**Aggregate-then-score**:
Combining errors over cells, times, components, and datasets before applying the score function once.
_Avoid_: score-then-average

**Local score**:
The score of the error in one cell, valid in a map but never averaged into a regional score.

**Overall Score**:
The total of one comparison against one dataset in one region: the score of $\sum_k w_k \mathrm{NMSE}_k$ over its score rows, where each orthogonal component of the error has weight 1, counted once, and every other row has weight 0.

**Variable score**:
The score of one variable: the weighted mean of the NMSEs of its datasets' Overall Scores, scored once.
By default each dataset is weighted by its number of observations.
Collier et al. (2018) call this aggregating scores across data sets for each variable.
_Avoid_: roll-up, aggregate score

**Section score**:
The score of one section: the weighted mean of the NMSEs of its variables, scored once, each variable weighted equally by default.
Regions overlap, so no score combines regions.
_Avoid_: roll-up, aggregate score

**Number of observations**:
The count of reference values in a region on the reference's own grid, $n_j$, the default weight of dataset $j$ in its variable score.
It depends only on the reference, so every model gets the same weights; within one dataset the cells and times are weighted by their area and length instead.

**Weight**:
The coefficient of one NMSE at one level: in the total, 1 for each orthogonal component and 0 for a diagnostic; in a variable or section score, a weighted mean whose weights sum to 1, with nested levels multiplying.
A weight that expresses the reliability of a dataset belongs in the error model; one that expresses importance is a value judgment.
Within an analysis, the area of a cell and the length of a time step are weights in the same sense.
_Avoid_: volume (for the coefficient of an orthogonal component)

**Scorecard**:
The display of an Overall Score with its component scores and diagnostics.

**Skill score**:
$1 - \mathrm{NMSE}$, the improvement of the comparison over the benchmark, unbounded below.

### Decomposition

**Component**:
A term of the total error: a normalized squared error that is added or averaged into an Overall Score, a variable score, or a section score, such as the error of one variable or of one dataset.
Components of different variables need not be orthogonal; a derived variable such as NEE shares error with its inputs.
_Avoid_: metric (for a score that enters no total)

**Orthogonal component**:
The part of the error in one subspace of an orthogonal decomposition, $P_k e$, such as the bias and the centered error of one variable.
The squared norms of orthogonal components sum to the squared error, with no cross terms.
Components of independent variables are orthogonal in the inner product weighted by the inverse reference covariance.
_Avoid_: component (when orthogonality matters)

**Four-term tree**:
The default decomposition into the mean bias, the pattern (the spatial variation of the period-mean bias), the seasonal term (the monthly climatology of the centered error), and the anomaly term (the rest).

**Bias term**:
An orthogonal component in which the reference has no variance about the benchmark, such as the mean bias.

**Part**:
A split of an orthogonal component along directions defined by the error itself, such as its amplitude and correlation parts; it shares its component's volume.

**Diagnostic**:
A score shown on the scorecard with weight 0 in the Overall Score and every score above it, because it is not an orthogonal component of the error or because it measures one already counted, such as the Taylor score, the phase shift, or a Time Series Bias alongside a Bias Score.
_Avoid_: component score

### Volumes

In this section, component means orthogonal component.

**Volume**:
An orthogonal component's share of the baseline's variance, $v_k$, and its coefficient in the average that gives back the NMSE; the volumes sum to 1.
_Avoid_: weight (a coefficient of a dataset or variable)

**Adjustment factor**:
The reciprocal of a component's volume, $1/v_k$.
_Avoid_: weight, multiplier

**Share**:
A component's squared error divided by the normalizer, $\lVert P_k e \rVert^2 / D$, an extensive error.
_Avoid_: contribution, unadjusted error

**Adjusted error**:
A component's share divided by its volume, $E_k^2$, an intensive error: the NMSE the comparison would have if its error in every component were like its error in this one.

**Baseline**:
The covariance $\rho$ of an error spread evenly, from which volumes are computed, such as white noise, a stationary process, or the realized reference.
_Avoid_: benchmark, null model

**Consistency in aggregation**:
The property that a total computed in stages equals the total computed at once, so that the ranking of models depends neither on the order or grouping in which components are combined nor on the score function.
_Avoid_: reassembly

### Design choices

**Derived choice**:
A design choice that follows from the rules of probability theory and a stated error model.

**Convention**:
A design choice that changes only how results are displayed, never which conclusions follow, such as $\alpha$.

**Modeling choice**:
A design choice that encodes prior information, such as the benchmark or the baseline.

**Value judgment**:
A design choice that expresses a scientific priority and enters as a utility, such as the importance of a variable.
