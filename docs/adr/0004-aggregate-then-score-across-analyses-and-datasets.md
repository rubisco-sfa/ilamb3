---
status: accepted
---

# The Overall Score and the roll-up average normalized MSEs, then score once

The Overall Score of a dataset and the dashboard roll-up across datasets
used to average scores,
so the ranking of models depended on the score function.
They now combine normalized mean squared errors (NMSEs)
and apply the score function once,
with the same mechanics used to combine the components of one error.
This extends [ADR 0002](0002-aggregate-then-score.md) from one analysis to every level of aggregation,
so that a ranking at any level does not depend on the score function.

## Context

Before this change,
`add_overall_score` took the unweighted mean of every score row for a model and region,
and `dataframe_to_cmec` in `ilamb3/meta.py` took unweighted means of the Overall Scores across datasets, variables, and sections
([how scores are combined now](../design/probabilistic-framework.md#how-scores-are-combined-now)).
A mean of scores reorders models when the score function changes.
In [fix 4](../design/probabilistic-framework.md#fixes-for-each-dataset),
averaging four component scores $e^{-\alpha \sqrt{\text{term}}}$ reorders 51 of 190 pairs of synthetic models between $\alpha = 0.25$ and $\alpha = 4$,
while scoring the summed error
gives the same order for every score function tried.
[Fix 14](../design/probabilistic-framework.md#fix-for-the-roll-up) proposes the same treatment for the roll-up,
and its weights and derived variables are listed among the [open questions](../design/probabilistic-framework.md#open-questions).

## Decision

Each component row of the results table carries its aggregated NMSE in a new `nmse` column,
and a label for the normalizer that the NMSE divides by in a new `normalizer` column.
Diagnostics carry no NMSE and enter neither the Overall Score nor any roll-up ([ADR 0005](0005-components-and-diagnostics.md)).

1. **Within a dataset.**
   `scoring.add_overall_score` in `ilamb3/analysis/scoring.py` sums,
   for each source and region,
   the NMSEs of the component rows that share a normalizer label.
   These are components of one error divided by one normalizer.
2. **Across normalizers.**
   It averages the sums for different labels with equal weights.
3. **Across datasets.**
   `dataframe_to_cmec` shows a single score as it is.
   For a group of datasets (the datasets of a variable, or of a section) it shows `scoring.score_nmse` of the mean of their NMSEs,
   with equal weight per dataset.
4. **Score once.**
   The score function is applied once,
   to the result of each step that is displayed,
   with the default method `Collier2018`, $s = e^{-\sqrt{\mathrm{NMSE}}}$.

Steps 2 and 3 are one operation:
a weighted mean of NMSEs with weights that sum to one,

$$\mathrm{NMSE} = \sum_j w_j \, \mathrm{NMSE}_j, \qquad \sum_j w_j = 1,
\qquad \text{score} = s(\mathrm{NMSE}).$$

The weights are equal by default.
A source and region with no component rows has no Overall Score.

## Justification

### 1. Components that share a normalizer add

The MSE is the squared bias plus the centered MSE ([Murphy, 1988](https://doi.org/10.1175/1520-0493(1988)116<2417:SSBOTM>2.0.CO;2), Eqs. 8–9; [Hodson et al., 2021](https://doi.org/10.1029/2021MS002681), Sec. 2).
Bias Score and RMSE Score both divide
by the pooled variance of the reference in time,
$D_{\mathrm{time}}$,
so without the uncertainty discount $\mathrm{NMSE} = \mathrm{NMSE}_{\mathrm{Bias}} + \mathrm{NMSE}_{\mathrm{RMSE}}$ ([fix 4](../design/probabilistic-framework.md#fixes-for-each-dataset)).
The sum adds the numerators of the two saved statistics
and keeps one denominator.
Concatenating the two sets of statistics and calling `scoring.aggregate` would count the denominator twice and halve the NMSE.

With $D = D_{\mathrm{time}}$ and no uncertainty discount,
$1 - \mathrm{NMSE}$ is the overall mean squared skill score against the benchmark ([WMO, 2007](https://www.metoffice.gov.uk/binaries/content/assets/metofficegovuk/pdf/research/climate-science/climate-observations-projections-and-impacts/svslrf.pdf), Sec. 3.3.1).
It has the form of the efficiency of [Nash and Sutcliffe (1970, Eq. 3)](https://doi.org/10.1016/0022-1694(70)90255-6),
$1 - \mathrm{RSS}/\mathrm{TSS}$,
the proportion of the initial variance accounted for by the model.
Below, $R^2_j = 1 - \mathrm{NMSE}_j$ denotes this skill score for group $j$.

### 2. Separately normalized groups combine by their volumes

Datasets, and groups with different normalizers, have no common normalizer,
so their MSEs cannot be summed (justification 4).
Each NMSE is dimensionless,
and the weighted mean $\sum_j w_j \mathrm{NMSE}_j = 1 - \sum_j w_j R^2_j$ is one minus the weighted mean skill score.

This is the volume formula of [volumes and adjustment factors](../design/probabilistic-framework.md#volumes-and-adjustment-factors),

$$\mathrm{NMSE} = \sum_k v_k E_k^2, \qquad \sum_k v_k = 1,
\qquad E_k^2 = \frac{\text{share}_k}{v_k},$$

applied to a decomposition by dataset.
Divide the error of each dataset $j$ by $\sqrt{D_j}$,
and weight dataset $j$ by $w_j$ in the mean over all data.
Every standardized reference has unit variance about its benchmark,
so dataset $j$ holds a fraction $w_j$ of the baseline's variance,
and its volume is $v_j = w_j$.
Its share is $w_j \mathrm{NMSE}_j$, the shares add to the total,
and its adjusted error is the share divided by the volume, $\mathrm{NMSE}_j$.

The roll-up is consistent in aggregation.
Nested weighted means with weights that sum to one at each level
equal one weighted mean with product weights:

$$\sum_g u_g \sum_{j \in g} w_{j \mid g} \,
\mathrm{NMSE}_j = \sum_j \big(u_{g(j)} \, w_{j \mid g(j)}\big) \,
\mathrm{NMSE}_j, \qquad \sum_g u_g = \sum_{j \in g} w_{j \mid g} = 1.$$

So a roll-up computed in stages equals the roll-up computed at once,
provided each level is computed from the NMSE of the level below
and never from its score.
With equal weight per dataset at every level, as in `dataframe_to_cmec`,
the section value is the mean of the variable values,
weighted by each variable's number of datasets.

### 3. The information in the NMSE

Let the reference be normal with variance $\sigma_x^2$,
and let the error of the comparison be unbiased, normal,
and independent of the comparison, with variance $\sigma_\epsilon^2$.
The differential entropy of a normal variable with variance $\sigma^2$ is $\tfrac12 \log(2\pi e \sigma^2)$
([Cover and Thomas, 1991](https://doi.org/10.1002/0471200611)),
so the mutual information between the reference and the comparison is

$$I = h(x) - h(x \mid f) = -\tfrac12 \log \frac{\sigma_\epsilon^2}{\sigma_x^2} = -\tfrac12 \log\left(1 - R^2\right) = -\tfrac12 \log \mathrm{NMSE}.$$

For one dataset,
$R^2$ and the NMSE are therefore monotone functions of the information,
and ranking by NMSE ranks by information.
[Cameron and Windmeijer (1997)](https://doi.org/10.1016/S0304-4076(96)01818-0) define $R^2$ for exponential-family regression as the proportionate reduction in uncertainty,
measured by Kullback–Leibler divergence, due to the regressors,
and show that for the normal distribution with known variance
it equals the usual coefficient of determination of the linear model.

The argument assumes an unbiased error.
With bias, the ranking by NMSE rests instead on the Gaussian likelihood
with an error variance fixed in proportion to the reference variance,
whose logarithm is linear in the NMSE ([the exponential of the NMSE](../design/probabilistic-framework.md#the-exponential-of-the-nmse)).

### 4. Normalize each MSE before combining

A sum of raw MSEs depends on the units of each variable,
so its ranking can change with units.
In [fix 11](../design/probabilistic-framework.md#fixes-for-other-analyses),
pooling raw Runoff Sensitivity errors ranks one model first
when the temperature sensitivity is in %/K
and the other first when it is in fractional change per K,
while the normalized error does not change.
Mutual information is invariant to rescaling each variable.
For invertible $\mathbf{A}$,
$h(\mathbf{A}\mathbf{X}) = h(\mathbf{X}) + \log\lvert\det\mathbf{A}\rvert$ ([Cover and Thomas, 1991](https://doi.org/10.1002/0471200611), Eq. 9.67),
and the log-determinants cancel in $I(\mathbf{X};\mathbf{Y}) = h(\mathbf{X}) + h(\mathbf{Y}) - h(\mathbf{X},\mathbf{Y})$.
[Kraskov, Stögbauer, and Grassberger (2004)](https://doi.org/10.1103/PhysRevE.69.066138) state the same invariance for any homeomorphisms of the marginal variables.
$R^2$ and the NMSE are also invariant to rescaling,
since the normalizer is in the squared units of the error.

### 5. The mean NMSE whitens independent variables

Standardizing the error of $m$ variables by the reference covariance $\boldsymbol\Sigma_x$
gives the mean squared Mahalanobis distance.
For $n$ times or cells,
with error vector $\mathbf{e}_t = \mathbf{f}_t - \mathbf{x}_t$,

$$1 - R^2_M = \frac{1}{nm} \sum_{t=1}^{n} \mathbf{e}_t^\top \boldsymbol\Sigma_x^{-1} \mathbf{e}_t .$$

For independent variables,
$\boldsymbol\Sigma_x = \operatorname{diag}(D_1, \dots, D_m)$,
where $D_j$ is the normalizer of variable $j$, so

$$1 - R^2_M = \frac{1}{m} \sum_{j=1}^{m} \frac{1}{D_j} \cdot \frac{1}{n}\sum_{t=1}^{n} e_{tj}^2 = \frac{1}{m}\sum_{j=1}^{m} \mathrm{NMSE}_j,
\qquad R^2_M = \frac1m \sum_{j=1}^m R^2_j .$$

The mean of the normalized MSEs is therefore the whitened MSE of independent variables,
and their errors are orthogonal components in the inner product weighted by $\boldsymbol\Sigma_x^{-1}$.
Replacing $1/m$ by weights $w_j$ that sum to one gives the weighted mean of the Decision.
The arithmetic mean is also proportional
to the negative log-likelihood ratio of the Gaussian error model,
with each error variance the same multiple $c$ of its reference variance
and the same number of data for each variable,
and it keeps the roll-up consistent in aggregation with the sum within a dataset:
by justification 2, the volume-weighted arithmetic mean of a dataset's adjusted component errors,
$\sum_k v_k E_k^2$, is that dataset's NMSE,
so splitting a dataset into its orthogonal components does not change the total.

For further consideration:
by justification 3, the information of $m$ independent variables is $-\tfrac12 \sum_j \log \mathrm{NMSE}_j$,
a function of the geometric mean of the NMSEs, not the arithmetic mean.
The two rank models differently when one dataset has a very small NMSE.
The volume-weighted geometric mean of the adjusted errors is not the NMSE,
$\sum_k v_k \log E_k^2 \ne \log \sum_k v_k E_k^2$,
so under it the total would change when a dataset is split into its orthogonal components.
This is why it is not adopted here.

### 6. Limits: dependent components

The mean assumes independent components.
For dependent components it counts shared error more than once:

- derived variables repeat the errors of their inputs;
  $e_{\mathrm{NEE}} = e_{\mathrm{Reco}} - e_{\mathrm{GPP}}$,
  so an error in GPP counts under GPP and again under NEE ([fix 14](../design/probabilistic-framework.md#fix-for-the-roll-up));
- references built from the same observations, such as FLUXCOM,
  which is upscaled from FLUXNET sites;
- analyses that measure overlapping parts of one error.

The general form whitens the errors by the full reference covariance,
$R^2_M$ with a non-diagonal $\boldsymbol\Sigma_x$.
When that covariance is estimated from few samples,
the shrinkage estimator of [Ledoit and Wolf (2004)](https://doi.org/10.1016/S0047-259X(03)00096-4),
the asymptotically optimal convex combination
of the sample covariance and a scaled identity under quadratic loss,
keeps its inverse stable.
This is not implemented.
[ADR 0005](0005-components-and-diagnostics.md) keeps overlapping metrics out of the total instead,
and the derived variables of the [roll-up across variables](../design/probabilistic-framework.md#roll-up-across-variables) are proposed as diagnostics.

### 7. Weights

Equal weights are a value judgment,
the importance of each group stated as a utility.
Weights derived from the reliability of each reference,
which belongs in the error model as a variance,
would be a derived choice ([the design rule](../design/probabilistic-framework.md#the-design-rule)).
`ilamb3` has neither kind of configurable weight;
ILAMB v2 weighted the RMSE score twice and weighted datasets and variables through the configuration ([Collier et al., 2018](https://doi.org/10.1029/2018MS001354), Sec. 2.4 and Eq. 31).

## Considered options

- **The mean of scores** (the previous behavior).
  Its ranking depends on the score function (fix 4).
- **The sum of raw MSEs.**
  It depends on the units of each variable (justification 4).
- **The mean of $E = \sqrt{\mathrm{NMSE}}$.**
  $E$ is not additive:
  $\sqrt{a + b} \ne \sqrt{a} + \sqrt{b}$,
  so the sum of the component $E$ is not the $E$ of the summed error,
  and the mean of $E$ is not the $E$ of any total.
- **The product of component scores.**
  It equals the score of the summed NMSE only for $s = e^{-\alpha\,\mathrm{NMSE}}$,
  the only continuous score with $s(0) = 1$ that turns sums into products ([the exponential of the NMSE](../design/probabilistic-framework.md#the-exponential-of-the-nmse); [ADR 0008](0008-exponential-score-function.md)).
  Under the current `Collier2018` function it does not.
  With the exponential score it is the same decision computed another way.
- **The geometric mean of NMSEs.**
  It is a function of the information of independent variables (justification 5),
  but the total would change when a dataset is split into its orthogonal components.
- **$R^2_M$ with a full covariance.**
  It handles dependent components (justification 6),
  but needs an estimated covariance across variables and datasets
  on common samples.
  It is left as future work.

## Consequences

- Overall Score values change for every dataset.
- A dataset whose analyses report only diagnostics, such as Area,
  Runoff Sensitivity, or a Relationship-only configuration,
  has no Overall Score.
- Diagnostics show no rolled-up value on the dashboard.
- Results tables written before this change have no `nmse` column,
  and `add_overall_score` gives them no Overall Score.
- Under `RegionalQuantiles`,
  the Overall Score and the roll-ups still apply the default score function,
  `Collier2018`.
  This is a convention.
  Bias under `RegionalQuantiles` divides by the squared quantile and has its own normalizer label,
  so its NMSE is averaged with the RMSE NMSE instead of added to it.
- For fixed weights,
  the ranking of models at every level is invariant to the score function,
  because each level applies one strictly decreasing function
  to a fixed weighted mean of NMSEs.

## References

- Cameron, A. C., & Windmeijer, F. A. G. (1997).
  An R-squared measure of goodness of fit
  for some common nonlinear regression models.
  *Journal of Econometrics*, 77(2), 329–342. https://doi.org/10.1016/S0304-4076(96)01818-0
- Collier, N., Hoffman, F. M., Lawrence, D. M., Keppel-Aleks, G., Koven, C. D.,
  Riley, W. J., Mu, M., & Randerson, J. T. (2018).
  The International Land Model Benchmarking (ILAMB) system:
  Design, theory, and implementation.
  *Journal of Advances in Modeling Earth Systems*, 10(11), 2731–2754. https://doi.org/10.1029/2018MS001354
- Cover, T. M., & Thomas, J. A. (1991).
  *Elements of Information Theory*.
  John Wiley & Sons. https://doi.org/10.1002/0471200611
- Hodson, T. O., Over, T. M., & Foks, S. S. (2021).
  Mean squared error, deconstructed.
  *Journal of Advances in Modeling Earth Systems*, 13, e2021MS002681. https://doi.org/10.1029/2021MS002681
- Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
  Estimating mutual information.
  *Physical Review E*, 69(6), 066138. https://doi.org/10.1103/PhysRevE.69.066138
- Ledoit, O., & Wolf, M. (2004).
  A well-conditioned estimator for large-dimensional covariance matrices.
  *Journal of Multivariate Analysis*, 88(2), 365–411. https://doi.org/10.1016/S0047-259X(03)00096-4
- Murphy, A. H. (1988).
  Skill scores based on the mean square error
  and their relationships to the correlation coefficient.
  *Monthly Weather Review*, 116(12), 2417–2424. https://doi.org/10.1175/1520-0493(1988)116<2417:SSBOTM>2.0.CO;2
- Nash, J. E., & Sutcliffe, J. V. (1970).
  River flow forecasting through conceptual models part I —
  A discussion of principles.
  *Journal of Hydrology*, 10(3), 282–290. https://doi.org/10.1016/0022-1694(70)90255-6
- World Meteorological Organization (2007).
  Standardised Verification System (SVS) for Long-Range Forecasts (LRF).
  *Manual on the Global Data-processing and Forecasting System* (WMO-No. 485),
  Attachment II.8.
  https://www.metoffice.gov.uk/binaries/content/assets/metofficegovuk/pdf/research/climate-science/climate-observations-projections-and-impacts/svslrf.pdf
