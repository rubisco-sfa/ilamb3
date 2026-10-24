# Formalizing ILAMB as a logical system

*Status:
proposal.* Aggregate-then-score within each analysis,
in the Overall Score, and in the roll-up is implemented on the `fix/aggregate-then-score` branch ([rubisco-sfa/ilamb3#217](https://github.com/rubisco-sfa/ilamb3/pull/217)).
The rest of this document has not been implemented.
The decisions are recorded in [`docs/adr/`](../adr/README.md).

## Abstract

Model evaluation is inference.
From observations that are incomplete and uncertain,
we judge how well a model represents the processes it is meant to represent,
and we act on that judgment: we choose among models, weight an ensemble,
or decide where a model needs work.
[Cox (1946)](https://doi.org/10.1119/1.1990764) showed that any rule for representing degrees of plausibility by real numbers must follow the product and sum rules of probability theory,
up to a monotone change of scale,
if it agrees with deductive logic when the premises are certain and gives the same result by every valid route of reasoning (under regularity conditions).
[Jaynes (2003, Chs. 1–2)](https://doi.org/10.1017/CBO9780511790423) developed this result into probability theory as the logic of plausible reasoning,
with deductive logic as its limiting case.
This document formalizes ILAMB as a logical system of that kind:
a framework for evaluating models whose rules for combining evidence are the rules of probability theory,
and in which every design choice is either derived from those rules or identified as a choice made for another reason.

The requirement that every valid route of reasoning give the same result takes a concrete form in model evaluation.
An ILAMB score combines errors over cells, times, components, variables,
and datasets, and they can be combined in many orders.
A framework is consistent in aggregation when every order gives the same result,
so that a total computed in stages equals the total computed at once.
In probability theory, this is the law of total expectation and,
at every level of a hierarchy, the tower property of conditional expectation.
In index-number theory, Vartia (1976) gave the requirement its name,
and [Diewert (1978)](https://doi.org/10.2307/1909755) studied which index formulas satisfy it.
In the measurement of poverty,
measures with the same property are called decomposable ([Foster, Greer, and Thorbecke, 1984](https://doi.org/10.2307/1913475)).
Under the rules proposed here, ILAMB is consistent in aggregation throughout:
the ranking of models does not depend on the score function used to display it,
and splitting one component of the error leaves the scores of the other components unchanged.

The squared error is aggregated before it is scored,
because it adds over the data as a log-likelihood does,
and the score is a monotone transformation of it.
The aggregated error is split into orthogonal components,
which are independent under a Gaussian error model and add without double counting;
metrics that are not components of the error are kept outside the total as diagnostics.
Each component is scored on its error per unit volume,
which is a conditional expectation under a probability measure on the components,
so the law of total expectation holds at every level of the decomposition,
and Gleason's theorem fixes the form of the measure.
The score $e^{-\alpha\,\mathrm{NMSE}}$ is a Gaussian likelihood ratio,
and the Gaussian is the distribution of maximum entropy for errors summarized by their mean square (Jaynes, 2003, Chs. 7 and 11).
Jaynes (2003, Ch. 20) treats the comparison of models as the comparison of their probabilities given the data.
With a stated error model,
a score of this form is the likelihood in that comparison;
with a conventional $\alpha$, it is a tempered version of it.

The same rules connect evaluation to uncertainty quantification.
Observational uncertainty, sampling variability,
and the structural error of a model are all expressed as probability distributions,
so a framework built on probability theory can carry them into its error model.
Probability theory does not make every choice,
and the framework states which choices it leaves open.
Some are conventions that change only how results are displayed,
such as $\alpha$, which sets the score of the benchmark.
Some are modeling choices that encode prior information,
such as the benchmark that defines the normalizer and the baseline covariance that defines what counts as error spread evenly.
Others are value judgments.
A weight that expresses how far a dataset is trusted is a statement about its uncertainty,
and it belongs in the error model as a variance.
A weight that expresses how much a variable, dataset,
or region matters is a scientific priority.
In decision theory, priorities enter as utilities,
which combine with probabilities to rank actions but are not derived from them (Jaynes, 2003, Chs. 13–14).
Such judgments are legitimate, and they are where a user's priorities belong.
The framework keeps them explicit and separate from the inference,
so that a reader can see which conclusions follow from the evidence and which follow from the priorities.

## The design rule

As far as practicable,
ILAMB should be consistent with and based on the rules of probability theory,
and those rules should be considered first in every design choice.
A choice that does not follow from them is still allowed,
but it is stated as one of these kinds:

- **Derived.** The choice follows from the rules of probability theory and a stated error model.
- **Convention.** The choice changes only how results are displayed,
  never which conclusions follow.
- **Modeling choice.** The choice encodes prior information,
  such as what counts as no skill.
- **Value judgment.** The choice expresses a scientific priority and enters as a utility.

| Choice | Kind | Basis | Section |
|---|---|---|---|
| Aggregate the squared error, then score | Derived | The squared error adds over the data, as a log-likelihood does; a ranking must not depend on a monotone transformation used for display | [Aggregating errors](#aggregating-errors-across-components) |
| Common units and normalizer for summed terms | Derived | Terms of one log-likelihood share one error scale | [Aggregating errors](#aggregating-errors-across-components) |
| Orthogonal components in the total, other metrics as diagnostics | Derived | Orthogonal components are independent under a Gaussian error model and add without double counting | [Decomposition](#the-orthogonal-decomposition-of-the-mse), [diagnostics](#components-and-diagnostics) |
| Component scores on the error per unit volume | Derived | Conditional expectation: the law of total expectation and the tower property; Gleason's theorem | [Volume](#the-volume-adjustment), [general rule](#a-general-rule-for-volumes), [factors](#volumes-and-adjustment-factors) |
| Score function $e^{-\alpha\,\mathrm{NMSE}}$ | Derived | Gaussian likelihood ratio; product rule for independent components; maximum entropy | [Score function](#the-score-function) |
| Error interval $b \pm 2 s_e$ | Derived | Chebyshev and Vysochanskij–Petunin bounds on the weighted error distribution | [Score function](#the-score-function) |
| $\alpha = \ln 2$ | Convention | Sets the benchmark's score to 0.5; does not change rankings | [Score function](#the-score-function) |
| Benchmark and baseline covariance $\rho$ | Modeling choice | What counts as no skill, and what counts as error spread evenly | [General rule](#a-general-rule-for-volumes) |
| Weights on datasets by reliability | Derived | The uncertainty of the reference belongs in the error model as a variance | [Abstract](#abstract) |
| Weights on variables, datasets, and regions by importance | Value judgment | Scientific priorities, stated explicitly as utilities | [Abstract](#abstract) |

The sections below follow the order in which the design was worked out:
what a sum of errors requires, how the MSE decomposes,
which metrics are components, how to score a component,
and which score function to use.

## Aggregating errors across components

An Overall Score that aggregates the component errors of a dataset applies the score function once to the total,
as [rubisco-sfa/ilamb3#217](https://github.com/rubisco-sfa/ilamb3/pull/217) does within each analysis.
A sum of errors is meaningful only if the components (a) are in the same units,
(b) are divided by the same normalizer, and (c) do not overlap.
This section lists where the current components do not yet meet these conditions and gives a fix for each,
with a synthetic experiment that checks it.
All 13 experiments pass;
they are in [`experiments/verify_fixes.py`](experiments/verify_fixes.py),
which prints one line per check, numbered to match the fixes.

### How scores are combined now

The list describes commit 5b4b215.
The working branch now aggregates the errors at levels 2 and 3 as well
([ADR 0004](../adr/0004-aggregate-then-score-across-analyses-and-datasets.md)):
each score row carries the NMSE it scores,
the Overall Score adds the NMSEs of the components that share a normalizer and scores the sum once,
and the roll-up scores the mean NMSE of the datasets.
Scores without an NMSE are diagnostics,
which neither enters.

1. **Within an analysis.** After [rubisco-sfa/ilamb3#217](https://github.com/rubisco-sfa/ilamb3/pull/217),
   the error is aggregated and then scored (Bias, RMSE, Hydro, nbp Trajectory, Runoff Sensitivity).
2. **Across the analyses for one dataset.** [`add_overall_score`](https://github.com/thodson-usgs/ilamb3/blob/5b4b2158ca1d3943e4a80f5fb89f22eec8a73075/ilamb3/analysis/base.py#L128-L142) takes the unweighted mean of every score row for a model and region.
   With the default analyses that is (Bias + RMSE + Seasonal Cycle + Spatial Distribution) / 4,
   plus one row for each configured relationship,
   so GPP against FLUXCOM-1 or WECANN-1-0 averages seven scores,
   three of them relationships.
3. **Across datasets, variables,
   and sections.** The dashboard roll-up in [`dataframe_to_cmec`](https://github.com/thodson-usgs/ilamb3/blob/5b4b2158ca1d3943e4a80f5fb89f22eec8a73075/ilamb3/meta.py#L86-L89) takes unweighted means of the Overall Scores.

ILAMB v2 also averaged scores at levels 2 and 3, but with weights:
the RMSE score counted twice ([Confrontation.py](https://github.com/rubisco-sfa/ILAMB/blob/8e58622c5a107327d7237554869c2c4e63633616/src/ILAMB/Confrontation.py#L292-L300); Collier et al., 2018, Eq. 31),
and datasets and variables were weighted through the configuration ([Scoreboard.py](https://github.com/rubisco-sfa/ILAMB/blob/8e58622c5a107327d7237554869c2c4e63633616/src/ILAMB/Scoreboard.py#L398-L405); Collier et al., 2018, Sec. 2.4). `ilamb3` has none of these weights.

### Units under aggregate-then-score

Averaging scores hides the units problem.
Every score is dimensionless, so an average of scores always has a value,
whatever each score measured.
A sum of errors does not: only quantities in the same units can be added,
and the sum ranks models consistently only if every term is divided by the same normalizer.
For a variable in units $[u]$,
each component must therefore become a mean square in $[u]^2$,
divided by one normalizer $D$ for the dataset.

Some measures are not in $[u]$ at all:
the Seasonal Cycle phase shift is in months,
the Spatial Distribution and Time Series Taylor scores combine a ratio and a correlation,
and Dispersion is a distance between probability distributions.
No normalizer puts these on the variable's scale,
so the fix is to replace each with the part of the mean squared error that it measures.
The other components are in $[u]$, or in another physical unit,
but are divided by different normalizers.

| Component | Measure now | Units | Normalizer now | Fix |
|---|---|---|---|---|
| Bias | \|period-mean bias\|, less the uncertainty | $[u]$ | reference standard deviation in time; the reference mean without a time dimension; or a regional quantile ([source](https://github.com/thodson-usgs/ilamb3/blob/5b4b2158ca1d3943e4a80f5fb89f22eec8a73075/ilamb3/analysis/bias.py#L276-L297)) | 1, 2, 5 |
| RMSE | centered RMSE, less the uncertainty | $[u]$ | the same standard deviation | 1, 3, 5 |
| Seasonal Cycle | \|phase shift\| | months | 6 months | 3 |
| Spatial Distribution | standard deviation ratio and correlation | none | none | 2 |
| Hydro | \|difference\| of the annual mean and of each season's mean | $[u]$ | standard deviation of annual means, or within the season ([source](https://github.com/thodson-usgs/ilamb3/blob/5b4b2158ca1d3943e4a80f5fb89f22eec8a73075/ilamb3/analysis/hydro.py#L42-L64)) | 7 |
| Dispersion | Hellinger distance | none | none | 8 |
| Time Series | difference of period means; Taylor score | $[u]$; none | \|reference mean\|; none | 9 |
| Area | missed area; excess area | area | reference area; comparison area | 10 |
| Runoff Sensitivity | \|sensitivity bias\|, less the uncertainty | %/% and %/K | none | 11 |
| Relationship | L2 difference of the binned mean responses | units of the dependent variable | L2 norm of the reference response | 12 |
| nbp | accumulated nbp outside the envelope; difference in one year | Pg C | uncertainty; or \|reference\| | 13 |

Bias and RMSE already meet all three conditions after [rubisco-sfa/ilamb3#217](https://github.com/rubisco-sfa/ilamb3/pull/217).
Both divide by the pooled variance of the reference in time,
$D_{\mathrm{time}} = \langle \overline{(x - \bar x)^2} \rangle$,
and the squared bias and the squared centered RMSE are orthogonal parts of the mean squared error (Murphy, 1988, Eqs. 8–9).
Without the uncertainty discount,
$E_{\mathrm{Bias}}^2 + E_{\mathrm{RMSE}}^2$ is the dataset's normalized MSE.
Spatial Distribution and Seasonal Cycle measure parts of these two terms,
so fixes 2 and 3 replace them with those parts instead of adding them.

**Notation.** $f$ is the comparison, $x$ the reference, and $e = f - x$.
An overbar is a time-weighted mean in one cell,
and $\langle \cdot \rangle$ is an area-weighted mean over the region.
$b = \bar e$ is the period-mean bias in a cell,
and $c = e - b$ is the centered error.

### Fixes for each dataset

**1. One normalizer.** Every term of a dataset's total must be divided by the same $D$.
Bias and RMSE use $D_{\mathrm{time}}$,
but Bias falls back to the reference mean for data without a time dimension,
such as biomass and soil carbon, which is a different scale.
The total variance

$$D_{\mathrm{total}} = \langle \overline{(x - \langle \bar x \rangle)^2} \rangle = D_{\mathrm{time}} + D_{\mathrm{space}},
\qquad D_{\mathrm{space}} = \langle (\bar x - \langle \bar x \rangle)^2 \rangle,$$

is defined with or without a time dimension;
Hodson et al. (2021, Sec. 3.2) normalize by this variance of all observations from all locations.
All models compared with one reference share $D$,
so the choice of $D$ does not change their ranking.
It sets the score values and each dataset's weight in the roll-up (fix 14).
With $D_{\mathrm{time}}$ and no uncertainty discount,
$1 - \mathrm{NMSE}$ is the overall mean squared skill score of the WMO standard for long-range forecasts (WMO, 2007, Sec. 3.3.1).

**Check (passed).** On the 10° grid of checks A–I, with 3.5 years of monthly values weighted with `ilamb3`'s `compute_cell_measures` and `compute_time_measures`,
$D_{\mathrm{total}}$ equals $D_{\mathrm{time}} + D_{\mathrm{space}}$ (5.21316 both ways).

**2. Spatial Distribution → the pattern term of the bias.** Spatial Distribution scores again part of the error that Bias already contains,
as a ratio and a correlation,
and without area weights ([source](https://github.com/thodson-usgs/ilamb3/blob/5b4b2158ca1d3943e4a80f5fb89f22eec8a73075/ilamb3/analysis/spatial_distribution.py#L127-L145)).
The pooled squared bias splits into the squared mean bias and the spatial variance of the bias:

$$\langle b^2 \rangle = \langle b \rangle^2 + \mathrm{Var}(b),
\qquad \mathrm{Var}(b) = \sigma_m^2 + \sigma_r^2 - 2 \sigma_m \sigma_r R,$$

where $\sigma_m$, $\sigma_r$,
and $R$ are the area-weighted spatial standard deviations and correlation of the period-mean fields (Gupta et al., 2009, Eq. 5, applied across space; the quantities of the Taylor diagram, Taylor, 2001).
$\mathrm{Var}(b)$ replaces the Spatial Distribution score inside the Bias term;
$\sigma_m / \sigma_r$ and $R$ remain diagnostics.
The proof,
and orthogonal splits of $\mathrm{Var}(b)$ into amplitude and correlation parts,
are in [the orthogonal decomposition of the MSE](#the-orthogonal-decomposition-of-the-mse).

**Check (passed).** A synthetic model has a mean bias, a random bias pattern,
and amplitude and phase errors.
Both identities hold to rounding with area weights (checks A and B).
The normalized standard deviation and correlation that Spatial Distribution reports,
which are not area-weighted,
give $\sigma_r^2 (\hat\sigma^2 + 1 - 2 \hat\sigma R)$,
which misses $\mathrm{Var}(b)$ by 2.7% on this grid.

**3. Seasonal Cycle → the seasonal term of the centered error.** The phase shift is in months and cannot be added to a mean square in $[u]^2$,
and the error it measures is already in the RMSE.
In each cell,
split the centered error into its time-weighted monthly climatology $c^{\mathrm{s}}$ and the anomalies $c^{\mathrm{a}} = c - c^{\mathrm{s}}$:

$$\langle \overline{c^2} \rangle = \langle \overline{(c^{\mathrm{s}})^2} \rangle + \langle \overline{(c^{\mathrm{a}})^2} \rangle.$$

The cross term vanishes because the anomalies have a time-weighted mean of zero within each calendar month.
This is a split by frequency, like the split into trend, seasonality,
and residual of Hodson et al. (2021, Sec. 3.4),
which uses STL and is only nearly orthogonal;
the climatology split is exactly orthogonal.
The seasonal term counts errors in amplitude as well as in phase.
The phase shift in months remains a diagnostic.
The proof,
and an orthogonal split of the seasonal term into amplitude and timing parts,
are in [the orthogonal decomposition of the MSE](#the-orthogonal-decomposition-of-the-mse).
The climatology must use the same time weights as the MSE for the cross term to vanish;
[`dataset.compute_monthly_climatology`](https://github.com/thodson-usgs/ilamb3/blob/5b4b2158ca1d3943e4a80f5fb89f22eec8a73075/ilamb3/dataset.py#L1355-L1361) currently returns its input unchanged.

**Check (passed).** The identity holds in every cell.
Shifting the reference seasonal cycle by 0 to 6 months, with no other error,
puts the error in the seasonal term: the anomaly term is 0,
the seasonal share is at least 98.59%,
and the seasonal term increases with the shift.
The rest, at most 1.41%,
is a period-mean bias,
because the grid's final year is incomplete.

**4. Score the total once.** With fixes 1–3, the dataset's error is

$$\mathrm{NMSE} = \frac{\langle b \rangle^2 + \mathrm{Var}(b) + \langle \overline{(c^{\mathrm{s}})^2} \rangle + \langle \overline{(c^{\mathrm{a}})^2} \rangle}{D},
\qquad \text{score} = s\big(\sqrt{\mathrm{NMSE}}\big).$$

The four terms share units and a normalizer, do not overlap,
and sum to the normalized MSE.
With $D = D_{\mathrm{time}}$,
the first two terms are the Bias statistics of [rubisco-sfa/ilamb3#217](https://github.com/rubisco-sfa/ilamb3/pull/217) and the last two are the RMSE statistics,
so $\mathrm{NMSE} = E_{\mathrm{Bias}}^2 + E_{\mathrm{RMSE}}^2$:
add the numerators of the two saved statistics and keep one denominator.
Concatenating the two sets of statistics and calling `scoring.aggregate` would count the denominator twice and halve the NMSE.
Each term's share of the total can be reported in place of the separate scores (Hodson et al., 2021, Fig. 1; Gupta et al., 2009, Eq. 6),
and the shares do not depend on $s$.

**Check (passed).** 20 synthetic models have random errors in mean bias,
bias pattern, amplitude, phase, noise, and correlation.
For every model, the four terms sum to the total.
Scoring the total ranks the models identically under $e^{-0.25E}$, $e^{-E}$,
$e^{-4E}$, and $1/(1+E)$.
Averaging the four component scores $e^{-\alpha \sqrt{\text{term}}}$ instead reorders 51 of the 190 pairs of models between $\alpha = 0.25$ and $\alpha = 4$.

**5. Apply the uncertainty once,
to the total.** Bias subtracts the uncertainty from $\lvert b \rvert$ and clips at 0,
and RMSE measures the distance outside the uncertainty interval at each time.
After either discount, the terms no longer sum to an MSE.
Apply the uncertainty once to the total, after the split, in one of two forms:

- **Error-variance correction.** If $\sigma_u$ is the standard error of the reference,
  and the reference error is independent of the truth and of the model,
  then the expected MSE against the reference is the MSE against the truth plus $\langle \overline{\sigma_u^2} \rangle$.
  The estimate $(\langle \overline{e^2} \rangle - \langle \overline{\sigma_u^2} \rangle) / D$ is unbiased for the normalized MSE against the truth.
- **Tolerance.** $\max(\langle \overline{e^2} \rangle - \langle \overline{u^2} \rangle, 0) / D$,
  which is closer to the current discount.

The shares of fix 4 are computed before the correction.
[`dataset.get_scalar_uncertainty`](https://github.com/thodson-usgs/ilamb3/blob/5b4b2158ca1d3943e4a80f5fb89f22eec8a73075/ilamb3/dataset.py#L1202-L1205) returns a standard deviation or standard error when the dataset supplies one.
For two bounds,
it returns the root sum of squares of their offsets from the value,
which is $\sqrt{2}\,h$ for symmetric bounds $\pm h$, not a standard deviation.

**Check (passed).** A reference has independent noise with a known $\sigma_u$ in each of 50 cells,
and the model is the truth.
Over 400 trials, the corrected NMSE averages 0.0001 (standard error 0.0002).
Subtracting keeps the terms additive.
Clipping each term separately, as now,
differs from clipping the total by a median of 33% across cells.

**6. Mass weighting:
the same weights in every term.** When mass weighting is enabled,
only Bias multiplies the area weights by the reference mean,
so Bias and RMSE are averaged with different weights.
Use the same weights in every term and in $D$;
the identities of fixes 1–4 hold for any positive weights.

**Check (passed).** With weights of area × |reference mean|,
the four terms still sum to the total, and the bias split still holds.

### Fixes for other analyses

**7. Hydro:
one error from the seasonal means.** The annual bias appears in all five Hydro scores,
which divide by the standard deviation of annual means or by the standard deviation within each season.
With season weights $p_s$ equal to each season's share of the time,

$$\sum_s p_s b_s = b, \qquad \sum_s p_s b_s^2 = b^2 + \sum_s p_s (b_s - b)^2.$$

Score $\langle b^2 + \sum_s p_s (b_s - b)^2 \rangle / D$ instead:
the annual bias and the seasonal spread of the bias, once each,
over one normalizer.
Where the full split of fix 4 is computed,
the seasonal spread is not added to it,
because each $b_s - b$ is the season's mean of $c^{\mathrm{s}}$,
so the spread is a coarser version of the seasonal term.

**Check (passed).** Both identities hold in every cell (the second is check K).
The seasonal spread (0.284 of $D$) is less than the seasonal term (0.359 of $D$),
as it must be for an average of $c^{\mathrm{s}}$.

**8. Dispersion → a second split of the centered error.** The Hellinger distance is on a probability scale and responds to differences in mean and spread,
which Bias and RMSE already contain.
In each cell,
sort both series and let $\omega = \mathrm{sort}(f) - \mathrm{sort}(x)$.
Then $\bar\omega = b$, and

$$\overline{c^2} = \mathrm{Var}(\omega) + \big[\,\overline{c^2} - \mathrm{Var}(\omega)\,\big]$$

are the distribution and sequence terms of Hodson et al. (2021, Sec. 3.5).
The sequence term is nonnegative because sorting both series maximizes their covariance.
This splits the same centered term as fix 3 in a different way,
so either split can be reported as a diagnostic,
but neither is added to the total.

**Check (passed).** In 2000 random pairs of series,
the means agree and the sequence term is nonnegative.
A time permutation of the reference has no bias or distribution error,
and a rescaling of the reference anomalies has no sequence error.

**9. Time Series:
one MSE per series.** The Time Series bias score divides the difference of the period means by the reference mean,
and its Taylor score measures the centered error as a ratio and a correlation,
so the two are on different scales.
They do not overlap: for a regional or site series,
the MSE splits into the squared bias and a centered term,
which is the Taylor quantity in time,

$$\overline{e^2} = \bar e^2 + \mathrm{Var}(e),
\qquad \mathrm{Var}(e) = \sigma_m^2 + \sigma_r^2 - 2 \sigma_m \sigma_r R,$$

Divided by the variance of the reference series,
the MSE gives one error in place of both scores.
$\sigma_m / \sigma_r$ and $R$ remain diagnostics.

**Check (passed).** The identity holds for the area-weighted regional mean series of the model in check 2.

**10. Area:
one error from the misclassified area.** The Missed Score is the overlap divided by the reference area,
and the Excess Score is the overlap divided by the comparison's own area ([source](https://github.com/thodson-usgs/ilamb3/blob/5b4b2158ca1d3943e4a80f5fb89f22eec8a73075/ilamb3/analysis/area.py#L160-L161)).
The second normalizer depends on the model, and the two scores are averaged.
With extent indicators $I$,

$$E^2 = \frac{\sum_i w_i (I_{m,i} - I_{r,i})^2}{\sum_i w_i I_{r,i}^2} = \frac{A_{\mathrm{missed}} + A_{\mathrm{excess}}}{A_{\mathrm{ref}}},$$

which is `scoring.statistics(I_m - I_r, I_r)` aggregated as in [rubisco-sfa/ilamb3#217](https://github.com/rubisco-sfa/ilamb3/pull/217).

**Check (passed).** The identity holds on the grid (check M).
With a reference area of 60, a model with an overlap of 50, 10 missed,
and 10 excess (20 misclassified) scores 0.833 as the mean of the two current scores.
A model with an overlap of 60, 0 missed,
and 22 excess (22 misclassified) scores higher,
at 0.866. The fixed error ranks the first model ahead (0.333 vs 0.367).

**11. Runoff Sensitivity:
normalize each sensitivity before pooling.** The normalizer is 1 ([source](https://github.com/thodson-usgs/ilamb3/blob/5b4b2158ca1d3943e4a80f5fb89f22eec8a73075/ilamb3/analysis/runoff_sensitivity.py#L119-L121)),
so the errors are in %/% for precipitation and %/K for temperature.
On [rubisco-sfa/ilamb3#217](https://github.com/rubisco-sfa/ilamb3/pull/217)'s branch,
the combined basin map adds their squares ([source](https://github.com/thodson-usgs/ilamb3/blob/5b4b2158ca1d3943e4a80f5fb89f22eec8a73075/ilamb3/analysis/runoff_sensitivity.py#L216-L222)),
and `main` averaged scores of the dimensional errors.
Divide each by a normalizer in its own units first:

$$E^2 = \tfrac{1}{2} \left( \frac{\langle e_P^2 \rangle}{D_P} + \frac{\langle e_T^2 \rangle}{D_T} \right),$$

with $D_P$ and $D_T$ such as the variance of each reference sensitivity across basins or its squared uncertainty.
The equal weights are a choice.

**Check (passed).** Two synthetic models are compared over 30 basins.
Pooling the raw errors ranks model A first when the temperature sensitivity is in %/K and model B first when it is in fractional change per K. The normalized error does not change with the units.

**12. Relationship:
center the responses.** The relative L2 difference of the binned mean responses ([source](https://github.com/thodson-usgs/ilamb3/blob/5b4b2158ca1d3943e4a80f5fb89f22eec8a73075/ilamb3/analysis/relationship.py#L199-L201)) responds to a pure bias in the dependent variable,
which Bias already contains:
a constant bias shifts the mean in every bin by the same amount.
Compare the centered responses instead,

$$E^2 = \frac{\operatorname{mean}_k \big(\Delta r_k - \overline{\Delta r}\big)^2}{D_{\mathrm{dep}}},$$

where $\Delta r_k$ is the difference in the mean response in bin $k$ and $D_{\mathrm{dep}}$ is a normalizer of the dependent variable.
Centering removes the overlap with the mean bias,
but not with the pattern and seasonal terms,
because the independent variable varies in space and time.
The relationship error can be reported beside the dataset's score or added to it with an explicit weight.

**Check (passed).** On synthetic data, using `ilamb3`'s `Relationship` class,
a bias of 0.5 in the dependent variable gives a current error of 0.106 and a centered error of 3 × 10⁻¹⁶.
A slope 20% too steep gives a centered error of 0.200,
normalized here by the root mean square of the centered reference response.

**13. nbp:
one error.** The Difference score scores the accumulated nbp in the evaluation year,
which is also one of the years that the Trajectory score pools,
with a different normalizer and $\alpha = \ln 2$ ([source](https://github.com/thodson-usgs/ilamb3/blob/5b4b2158ca1d3943e4a80f5fb89f22eec8a73075/ilamb3/analysis/nbp.py#L161-L187)).
Score the Trajectory error and report the difference in the evaluation year as a scalar.
If the final year should count more,
give it an explicit weight in the pooled sums.

**Check (not run).** The overlap holds by construction whenever the evaluation year is one of the pooled years.

### Fix for the roll-up

**14. Pool the normalized errors with explicit weights.** Each dataset's NMSE is dimensionless,
a fraction of its reference variance,
so datasets and variables can be pooled without a units problem:

$$E^2 = \frac{\sum_d \omega_d \, \mathrm{NMSE}_d}{\sum_d \omega_d},
\qquad \text{score} = s(E).$$

For fixed weights $\omega_d$, the ranking is invariant to $s$.
Two problems remain.
First, the weights are a judgment;
the current nested unweighted means are one choice.
Second, derived variables repeat the errors of their inputs.
For example, $e_{\mathrm{NEE}} = e_{\mathrm{Reco}} - e_{\mathrm{GPP}}$, so

$$\overline{e_{\mathrm{NEE}}^2} = \overline{e_{\mathrm{Reco}}^2} + \overline{e_{\mathrm{GPP}}^2} - 2\,\overline{e_{\mathrm{Reco}}\,e_{\mathrm{GPP}}},$$

and an error in GPP counts once under GPP and again under NEE.
The derived variables in [`ilamb3/configure/ilamb.yaml`](https://github.com/thodson-usgs/ilamb3/blob/5b4b2158ca1d3943e4a80f5fb89f22eec8a73075/ilamb3/configure/ilamb.yaml) are:

- **Carbon:** NEE = ra + rh − gpp and ecosystem respiration = ra + rh,
  alongside GPP.
- **Hydrology:** evapotranspiration and latent heat,
  which are the same flux up to the latent heat of vaporization;
  evaporative fraction = hfls / (hfls + hfss),
  alongside latent and sensible heat.
- **Radiation:** net shortwave = rsds − rsus, net longwave = rlds − rlus,
  net radiation (their sum), and albedo = rsus / rsds,
  alongside upward shortwave and longwave and, under Forcings,
  downward shortwave and longwave.
- **Forcings:** diurnal temperature range = tasmax − tasmin,
  alongside tasmax and tasmin.

Leave the derived variables out of the roll-up and report them as diagnostics,
or set their weights explicitly.
The references for one variable are not independent either; for example,
FLUXCOM is upscaled from FLUXNET sites.
Averaging the NMSE over the references for a variable before the roll-up gives each variable one weight.

**Check (passed).** With random GPP and respiration errors,
the NEE identity holds, and an error in GPP alone gives NEE the same MSE as GPP.

### Open questions

1. Normalizer: $D_{\mathrm{time}}$,
   which matches the WMO skill score and [rubisco-sfa/ilamb3#217](https://github.com/rubisco-sfa/ilamb3/pull/217)'s Bias and RMSE,
   or $D_{\mathrm{total}}$, which is defined without a time dimension?
2. Score function for the total: $e^{-E}$ with $E = \sqrt{\mathrm{NMSE}}$,
   or $e^{-\alpha \cdot \mathrm{NMSE}}$ (Hodson et al., 2021, Eq. 15),
   under which the component scores multiply to the total score?
   The ranking is the same either way.
   [The score function](#the-score-function) recommends the second.
3. Uncertainty: error-variance correction or tolerance?
   How should interval bounds be converted to a standard deviation?
4. Relationships: report them beside the Overall Score,
   or include them with an explicit weight?
5. Roll-up: which derived variables to leave out, and which weights to use?

## The orthogonal decomposition of the MSE

The errors that the Spatial Distribution and Seasonal Cycle scores target are exact orthogonal components of the MSE.
The two scores themselves are not components:
each is a different function of the same statistics,
and each can rank two models in the opposite order to the component it targets.
The components can therefore be aggregated to give back the MSE,
but the scores cannot,
which is why fixes 2 and 3 in [Aggregating errors across components](#aggregating-errors-across-components) replace them.
This section gives the proofs and splits each component further,
into an amplitude part and a correlation or timing part.
Nine numerical checks (A–I), summarized at the end of the section,
confirm each identity.

**Notation**,
as in [Aggregating errors across components](#aggregating-errors-across-components):
$f$ is the comparison, $x$ the reference, and $e = f - x$.
An overbar is a time-weighted mean in one cell, with time weights $\tau_t$,
and $\langle \cdot \rangle$ is an area-weighted mean over the region,
with area weights $w_i$.
$b = \bar e$ and $c = e - b$;
$c^{\mathrm{s}}$ is the time-weighted monthly climatology of $c$,
and $c^{\mathrm{a}} = c - c^{\mathrm{s}}$.
Primes denote period-mean fields with their regional mean removed,
$m' = \bar f - \langle \bar f \rangle$ and $r' = \bar x - \langle \bar x \rangle$.
$\sigma_m$, $\sigma_r$,
and $R$ are their area-weighted standard deviations and correlation,
and $\hat\sigma = \sigma_m / \sigma_r$.

### Four orthogonal terms

$(g, h) = \langle \overline{g h} \rangle$ is an inner product on fields over the region and period,
and $(e, e)$ is the MSE.
Consider the nested subspaces

$$V_0 \subset V_1 \subset V_2 \subset V_3,$$

where $V_0$ holds the constants, $V_1$ the fields that are constant in time,
$V_2$ the fields that depend only on the cell and the calendar month,
and $V_3$ all fields.
A weighted mean is a least-squares fit,
so the orthogonal projections of $e$ onto these subspaces are

$$P_0 e = \langle b \rangle, \qquad P_1 e = b,
\qquad P_2 e = b + c^{\mathrm{s}}, \qquad P_3 e = e.$$

Because the subspaces are nested, $P_{k-1} P_k = P_{k-1}$.
Let $P_{-1} = 0$.
For $j < k$, the increment $(P_j - P_{j-1}) e$ lies in $V_j \subseteq V_{k-1}$,
while $(P_k - P_{k-1}) e = (I - P_{k-1}) P_k e$ is orthogonal to $V_{k-1}$.
The four increments are therefore mutually orthogonal, and by Pythagoras

$$\mathrm{MSE} = \underbrace{\langle b \rangle^2}_{\text{mean bias}} + \underbrace{\langle (b - \langle b \rangle)^2 \rangle}_{\text{pattern}} + \underbrace{\langle \overline{(c^{\mathrm{s}})^2} \rangle}_{\text{seasonal}} + \underbrace{\langle \overline{(c^{\mathrm{a}})^2} \rangle}_{\text{anomaly}}.$$

This is the structure of an analysis of variance.
The proof needs product weights $w_i \tau_t$ and a climatology that uses the same time weights as the MSE.
With missing values,
$\langle b \rangle$ must weight each cell by its area times its valid time.
Complete years are not needed (check A).

### Spatial Distribution

The pattern term is the spatial error of the period means,
$b - \langle b \rangle = m' - r'$.
Expanding the square gives the law of cosines of the Taylor diagram (Taylor, 2001):

$$\langle (b - \langle b \rangle)^2 \rangle = \sigma_m^2 + \sigma_r^2 - 2 \sigma_m \sigma_r R.$$

The Spatial Distribution score is a different function of the same statistics ([source](https://github.com/thodson-usgs/ilamb3/blob/5b4b2158ca1d3943e4a80f5fb89f22eec8a73075/ilamb3/analysis/spatial_distribution.py#L128-L145)):

$$S = \frac{2 (1 + R)}{(\hat\sigma + 1/\hat\sigma)^2}.$$

$S$ is not a function of the pattern term,
so it is not a component of the MSE (check D):

| Model | $\hat\sigma$ | $R$ | Pattern term / $\sigma_r^2$ | $S$ |
|---|---|---|---|---|
| 1 | 0.5 | 1 | 0.25 | 0.64 |
| 2 | 1 | 0.8 | 0.40 | 0.90 |

$S$ ranks model 2 first although its pattern error is 60% larger.
For fixed $R$,
the pattern term divided by $\sigma_r^2$ is $\hat\sigma^2 + 1 - 2 \hat\sigma R$,
which is smallest at $\hat\sigma = R$:
a model that is imperfectly correlated lowers its MSE by damping its pattern.
$S$ is largest at $\hat\sigma = 1$ for every $R$.
The law of cosines also needs area weights and the same cells for every statistic,
and Spatial Distribution uses neither.

The pattern term splits further into an amplitude part and a correlation part.
Each of three splits sums to the pattern term (checks B and C):

| Split | Amplitude part | Correlation part | Orthogonal |
|---|---|---|---|
| Bisector | $\tfrac12 (\sigma_m - \sigma_r)^2 (1 + R)$ | $\tfrac12 (\sigma_m + \sigma_r)^2 (1 - R)$ | yes |
| Murphy (1988) | $(\sigma_m - R \sigma_r)^2$ | $\sigma_r^2 (1 - R^2)$ | yes |
| Gupta et al. (2009) | $(\sigma_m - \sigma_r)^2$ | $2 \sigma_m \sigma_r (1 - R)$ | no |

For the bisector split, let $u = r' / \sigma_r$ and $v = m' / \sigma_m$,
which have unit norm.
Then

$$m' - r' = \tfrac12 (\sigma_m - \sigma_r)(u + v) + \tfrac12 (\sigma_m + \sigma_r)(v - u),$$

and $(u + v, v - u) = \lvert v \rvert^2 - \lvert u \rvert^2 = 0$.
With $\lvert u + v \rvert^2 = 2 (1 + R)$ and $\lvert v - u \rvert^2 = 2 (1 - R)$,
the squared norms of the two parts are the entries in the table.
The bisector split is the only one of the three that is symmetric in model and reference:
its amplitude part is zero exactly when $\sigma_m = \sigma_r$,
and its correlation part exactly when $R = 1$.
Murphy's split projects the reference pattern onto the model pattern,
so its amplitude part, the conditional bias,
is zero at $\hat\sigma = R$ instead.
Gupta's terms are nonnegative and sum to the pattern term,
but the vectors behind them, $(\sigma_m - \sigma_r) v$ and $\sigma_r (v - u)$,
are not orthogonal.

Gupta's split is the bias, distribution,
and sequence split of Hodson et al. (2021, Sec. 3.5), applied across space,
when the two fields have distributions of the same shape.
In general,
let $Q_m$ and $Q_r$ be the area-weighted quantile functions of $\bar f$ and $\bar x$,
and let $R_q$ be their correlation over $p \in [0, 1]$.
Then

$$\langle (b - \langle b \rangle)^2 \rangle = \underbrace{(\sigma_m - \sigma_r)^2}_{\text{scale}} + \underbrace{2 \sigma_m \sigma_r (1 - R_q)}_{\text{shape}} + \underbrace{2 \sigma_m \sigma_r (R_q - R)}_{\text{sequence}},
\qquad R \le R_q \le 1.$$

The scale and shape terms sum to the variance of $Q_m - Q_r$,
and adding the squared mean bias gives $\int_0^1 (Q_m - Q_r)^2 \, dp$.
That integral is the squared 2-Wasserstein distance between the two distributions of period means,
the smallest MSE over all pairings of their values.
The pairing by cell is one such pairing, so the sequence term,
the error that comes from where the model places its values, is nonnegative,
which proves $R \le R_q$; $R_q \le 1$ by the Cauchy–Schwarz inequality.

### Seasonal Cycle

The climatology is linear,
so the seasonal error is the difference of the two mean annual cycles,
$c^{\mathrm{s}} = s_m - s_r$,
where $s_m$ and $s_r$ are the climatologies of $f$ and $x$ with their period means removed.
In each cell,
with $\sigma_{s,m}$ and $\sigma_{s,r}$ the standard deviations of the two cycles (their amplitudes) and $R_s$ their correlation across calendar months,
the same law of cosines holds (check E):

$$\overline{(c^{\mathrm{s}})^2} = \sigma_{s,m}^2 + \sigma_{s,r}^2 - 2 \sigma_{s,m} \sigma_{s,r} R_s.$$

For sinusoidal cycles $a_m \sin(\omega t - \phi)$ and $a_r \sin(\omega t)$,
with a lag of $\Delta$ months so that $\phi = 2 \pi \Delta / 12$,
the variances are $a_m^2 / 2$ and $a_r^2 / 2$ and the covariance is $\tfrac12 a_m a_r \cos\phi$,
so

$$R_s = \cos \frac{2 \pi \Delta}{12}$$

and the phase enters the MSE through $1 - \cos\phi$ (check F).
The bisector split, applied in each cell,
gives an amplitude part and a timing part:

$$\overline{(c^{\mathrm{s}})^2} = \underbrace{\tfrac12 (\sigma_{s,m} - \sigma_{s,r})^2 (1 + R_s)}_{\text{amplitude}} + \underbrace{\tfrac12 (\sigma_{s,m} + \sigma_{s,r})^2 (1 - R_s)}_{\text{timing}}.$$

For sinusoids,
the timing part is $\tfrac12 (a_m + a_r)^2 \sin^2(\pi \Delta / 12)$:
zero at no lag, largest at a lag of 6 months,
and proportional to the squared sum of the amplitudes.
The area-weighted sums of the two parts remain orthogonal,
because parts in different cells have disjoint support.
For cycles with several harmonics,
Parseval's theorem splits the seasonal term by harmonic (exactly, with equal month weights),
and each harmonic splits the same way.
With complex amplitudes $A_m$ and $A_r$ and phase difference $\phi$,

$$\lvert A_m - A_r \rvert^2 = (\lvert A_m \rvert - \lvert A_r \rvert)^2 \cos^2 \tfrac{\phi}{2} + (\lvert A_m \rvert + \lvert A_r \rvert)^2 \sin^2 \tfrac{\phi}{2}.$$

The Seasonal Cycle score, $1 - \lvert \Delta \rvert / 6$,
uses only the difference in the month of the peak,
with no amplitude ([source](https://github.com/thodson-usgs/ilamb3/blob/5b4b2158ca1d3943e4a80f5fb89f22eec8a73075/ilamb3/analysis/cycle.py#L73-L88)),
so it is not a function of the seasonal term (check G):

| Model | Amplitude / reference | Lag (months) | Seasonal term / reference cycle variance | Shift score |
|---|---|---|---|---|
| A | 2.5 | 0 | 2.25 | 1.00 |
| B | 1 | 2 | 1.00 | 0.67 |
| C | 2 | 0 | 1.00 | 1.00 |

B and C have the same seasonal error but different scores,
and A has more than twice B's seasonal error but the higher score.
Cells with almost no seasonal cycle, where the peak month is set by noise,
count fully in the regional mean shift but contribute almost nothing to the timing part.

### The full decomposition

$$\mathrm{MSE} = \langle b \rangle^2 + \underbrace{A_{\mathrm{pat}} + C_{\mathrm{pat}}}_{\text{pattern}} + \underbrace{A_{\mathrm{seas}} + T_{\mathrm{seas}}}_{\text{seasonal}} + \langle \overline{(c^{\mathrm{a}})^2} \rangle,$$

where $A$, $C$, and $T$ are the amplitude, correlation,
and timing parts of the bisector splits.
The six terms are mutually orthogonal:
the pattern parts lie in $V_1 \cap V_0^\perp$,
the seasonal parts in $V_2 \cap V_1^\perp$,
and each pair is orthogonal within its subspace (check H).
Divided by one normalizer, the total is scored once,
as in fix 4. Each term's share of the total describes the same aspects of the error that Spatial Distribution and Seasonal Cycle describe now,
on the scale of the MSE.

What the MSE cannot keep is the Taylor score's preference for matching variance.
Weighting the two bisector parts of the pattern term by $\alpha$ and $\beta$ and setting the derivative in $\hat\sigma$ to zero gives,
for fixed $R$, the optimum

$$\hat\sigma^* = \frac{\alpha (1 + R) - \beta (1 - R)}{\alpha (1 + R) + \beta (1 - R)}.$$

With equal weights, the only weights under which the parts sum to the MSE,
the optimum is $R$;
it reaches 1 only if the correlation part is dropped ($\beta = 0$) (check I).
A variance mismatch can be reported,
for example as the scale term or the bisector amplitude part,
but weighting it more than the MSE does breaks the sum.
This is a choice between two criteria, not a derivation that is missing.

### Checks

Synthetic data on a 10° grid with 3.5 years of monthly values,
so the final year is incomplete,
weighted with `ilamb3`'s `compute_cell_measures` and `compute_time_measures`.
All nine checks pass.

| Check | Result |
|---|---|
| A. Four-term split | The six cross terms are at most 7 × 10⁻¹⁷ of the MSE, and the four terms sum to it. The unweighted monthly mean that Seasonal Cycle uses leaves a cross term of 10⁻⁶ of the MSE. |
| B. Pattern splits | The law of cosines holds; the bisector and Murphy splits are orthogonal, with the parts in the table. Gupta's two vectors have a cross term of −35.5% of the pattern term. |
| C. Scale, shape, sequence | Area-weighted $R = 0.807 \le R_q = 0.997 \le 1$, and the distribution term matches. A shuffled affine copy of a skewed field has $R_q = 1$, which reduces the split to Gupta's. |
| D. Taylor score | The two models in the table. For $R = 0.6$, the pattern term is smallest at $\hat\sigma = 0.60$ and $S$ is largest at $\hat\sigma = 1.00$. |
| E. Seasonal law of cosines | $c^{\mathrm{s}} = s_m - s_r$, and the law of cosines holds in every cell. The amplitude and timing parts are 14.9% and 85.1% of the seasonal term and are orthogonal. |
| F. Sinusoids and harmonics | $R_s = \cos(2 \pi \Delta / 12)$ to within 2 × 10⁻¹⁶ with equal month weights, and 4 × 10⁻³ with month lengths as weights. The per-harmonic parts of a two-harmonic cycle sum to its seasonal term. |
| G. Phase-shift score | The three models in the table. |
| H. Full decomposition | The six terms sum to the MSE to rounding: mean bias 68.8%, pattern amplitude 0.6%, pattern correlation 1.6%, seasonal amplitude 3.2%, seasonal timing 18.4%, anomaly 7.3%. |
| I. Weighted optimum | For $R = 0.6$, the optimal $\hat\sigma$ for weights $(\alpha, \beta)$ of (1, 1), (2, 1), (5, 1), and (1, 0) is 0.600, 0.778, 0.905, and 1.000, as the formula gives. |

The checks are in [`experiments/verify_components.py`](experiments/verify_components.py).

## Components and diagnostics

Make the orthogonal decomposition of the MSE the default:
one total score per dataset,
with its orthogonal components shown as shares of the total.
Move the canonical ILAMB metrics into a separate suite of diagnostics.
They are still computed and shown on the scorecard,
but they are visually distinct and excluded from the total and from every roll-up of it.
For most canonical metrics,
an orthogonal component captures essentially the same concept,
so the default gives up little.
Checks J–N extend checks A–I of [the previous section](#the-orthogonal-decomposition-of-the-mse) on the same synthetic data.

### Rule

A metric contributes to the total only if it is one term of a single orthogonal decomposition of the MSE,
counted once.
Two kinds of metric fail this rule:

- **Metrics that are not components of the MSE.** The Taylor score,
  the phase shift, the Hellinger distance,
  and the relationship error each apply a different measure to statistics that overlap the MSE.
- **Components that are already in the total.** Some metrics are orthogonal pieces,
  or coarser versions, of terms that the total already contains.
  Adding them counts that error twice.

### Default decomposition

The anomaly term splits the same way as the pattern and seasonal terms.
In each cell, $c^{\mathrm{a}} = a_m - a_r$,
the difference between the model's and the reference's anomalies from their own monthly climatologies.
With $\sigma_{a,m}$ and $\sigma_{a,r}$ the standard deviations of the two anomaly series and $R_a$ their correlation,
the bisector split gives an amplitude part $\tfrac12 (\sigma_{a,m} - \sigma_{a,r})^2 (1 + R_a)$ and a correlation part $\tfrac12 (\sigma_{a,m} + \sigma_{a,r})^2 (1 - R_a)$.
Both lie in $V_3 \cap V_2^\perp$,
so the default decomposition has seven mutually orthogonal terms (check J):

$$\mathrm{MSE} = \langle b \rangle^2 + \underbrace{A_{\mathrm{pat}} + C_{\mathrm{pat}}}_{\text{pattern}} + \underbrace{A_{\mathrm{seas}} + T_{\mathrm{seas}}}_{\text{seasonal}} + \underbrace{A_{\mathrm{anom}} + C_{\mathrm{anom}}}_{\text{anomaly}}.$$

Parts from two different splits of the same term are not orthogonal to each other in general,
so the default uses one split per term and reports the others as diagnostics:

- the seasonal term into season means and a within-season remainder,
  instead of amplitude and timing (check K);
- the seasonal and anomaly terms into spatially uniform and local parts (check L);
- the centered term into Hodson's distribution and sequence terms,
  which are additive but not orthogonal.

### Canonical metrics and their orthogonal counterparts

Bias and RMSE are already orthogonal:
their statistics are the two halves of the total,
and their scores move to the suite as scores of those halves.
Every other canonical metric moves to the suite.
The default replaces each one with the orthogonal component in the last two columns where one exists.

| Canonical metric | Concept | Orthogonal counterpart in the default | Change needed |
|---|---|---|---|
| Bias Score | Period-mean bias | Mean bias + pattern | None (fixes 1, 5, and 6 apply) |
| RMSE Score | Error after removing the bias | Seasonal + anomaly | None (fixes 1, 5, and 6 apply) |
| Spatial Distribution | Spatial variability and pattern correlation | Pattern amplitude, pattern correlation | Area weights, common cells, MSE scale |
| Seasonal Cycle | Timing of the annual cycle | Seasonal timing (with seasonal amplitude) | Timing part in place of the peak-month difference |
| Dispersion | Distribution of values in time | Seasonal and anomaly amplitude parts | Captures the spread, not the shape |
| Hydro, annual | Annual-mean bias | The bias term | None |
| Hydro, seasonal | Seasonal biases | Season-mean part of the seasonal term (an alternative split) | One normalizer |
| Time Series of a gridded dataset's regional mean | Bias and Taylor statistics of the regional series | Mean bias + uniform parts of the seasonal and anomaly terms (an alternative split) | None; already in the total |
| Time Series as a dataset's only analysis | Bias and Taylor statistics of the series | The series' mean bias, amplitude, and correlation parts | Normalize by the variance of the series |
| Area, Missed and Excess | Missed and excess extent | Missed area and excess area, each over the reference area | Excess divides by the reference area, not the comparison's |
| nbp Difference | Accumulated carbon in the evaluation year | That year's share of the Trajectory error | Share of the pooled sum |
| Runoff Sensitivity | Precipitation and temperature sensitivity | Each sensitivity's error, normalized | A normalizer in each sensitivity's units |
| Relationship | Response to a driver | None | Stays in the suite |

Notes on the rows:

- **Spatial Distribution.** The pattern amplitude part is zero exactly when $\sigma_m = \sigma_r$ and,
  for fixed $R$, smallest there.
  It therefore keeps the Taylor score's preference for matching variance within that part;
  only the total prefers $\hat\sigma = R$.
- **Seasonal Cycle.** For sinusoids,
  the timing part is $\tfrac12 (a_m + a_r)^2 \sin^2(\pi \Delta / 12)$.
  Like the shift score, it increases with the lag up to 6 months,
  but it is weighted by the cycle amplitudes,
  so cells with almost no cycle count for little.
- **Dispersion.** The Hellinger distance compares whole distributions.
  The amplitude parts measure their spread;
  no orthogonal component measures their shape.
- **Hydro,
  seasonal.** The time-invariant fields lie within the functions of cell and season,
  which lie within the functions of cell and month.
  These nested subspaces split the seasonal term orthogonally into $\sum_s p_s (b_s - b)^2$,
  the spread of Hydro's seasonal biases about the annual bias,
  and a within-season remainder (check K).
  This is an alternative to the amplitude and timing split.
- **Time Series.** The MSE of a regional-mean series is the mean bias plus the spatially uniform parts of the seasonal and anomaly terms (check L),
  so for a gridded dataset it is already in the total.
- **Area.** Missed and excess cells are disjoint,
  so the two errors are orthogonal (check M).
  The concepts are unchanged; only the Excess normalizer changes.
- **nbp.** Years are disjoint,
  so the yearly terms of the pooled Trajectory error are orthogonal (Hodson et al., 2021, Sec. 3.3, for components that are disjoint in time).
  The Difference score measures the raw difference in the evaluation year;
  that year's share measures the exceedance outside the envelope.
- **Relationship.** The error explained by binned values of a driver is an orthogonal split of the MSE (the law of total variance).
  It conditions both sources on the same observed driver, however,
  whereas a relationship compares each source's response to its own driver.
  It also cuts across the tree:
  in check N it captures all of the mean bias term but only 15% of the pattern term and 5% of the anomaly term.
  It could serve as an alternative breakdown,
  but it does not measure the relationship concept,
  so the relationship scores stay in the suite.

### Roll-up across variables

The same rule applies to the roll-up:
the total should cover a set of variables none of which is a function of the others.
The derived variables move to the suite:

- net shortwave, net longwave, net radiation, and albedo,
  keeping the four upward and downward fluxes;
- evaporative fraction, keeping latent and sensible heat;
- diurnal temperature range, keeping the maximum and minimum temperatures;
- one of evapotranspiration and latent heat;
- one of NEE and ecosystem respiration.

Which carbon variable counts as derived is a choice.
In flux-tower references,
GPP and ecosystem respiration are partitioned from the measured NEE.

### Scorecard

A sketch, with placeholder values:

```
GPP · FLUXCOM-1                                   Overall score  0.xx
In the total: shares of the normalized MSE (sum to 100%)
  Mean bias               ██████████░░░░░░░░░░  xx%
  Pattern    amplitude    █░░░░░░░░░░░░░░░░░░░  xx%
             correlation  ███░░░░░░░░░░░░░░░░░  xx%
  Seasonal   amplitude    ██░░░░░░░░░░░░░░░░░░  xx%
             timing       ████░░░░░░░░░░░░░░░░  xx%
  Anomaly    amplitude    █░░░░░░░░░░░░░░░░░░░  xx%
             correlation  █████░░░░░░░░░░░░░░░  xx%
Canonical metrics: not in the total
  Bias Score 0.xx · RMSE Score 0.xx
  Spatial Distribution 0.xx (σ̂ x.xx, R x.xx)
  Seasonal Cycle 0.xx (shift x.x months)
  Relationships: pr 0.xx · tas 0.xx · rsds 0.xx
```

The shares sum to the total and share one visual style.
The canonical metrics sit in a separate, muted group,
and no roll-up of the total uses them.

### Implementation

Each score row in the results table has an `nmse` column,
the aggregated NMSE that it scores,
and a `normalizer` label.
`scoring.add_overall_score` sums the NMSEs that share a label,
averages the sums for different labels,
and scores the result once;
rows without an NMSE are diagnostics and are left out.
The dashboard roll-up in `meta.dataframe_to_cmec` scores the mean NMSE of the datasets and does not roll up diagnostics.
Diagnostics keep `type == "score"`, so the dashboard still shows them;
a separate type, such as `"diagnostic"`,
would let the scorecard style them apart.

### Checks

Same synthetic data as checks A–I. All five checks pass.

| Check | Result |
|---|---|
| J. Seven-term default | Anomaly error = model anomalies − reference anomalies. The largest of the 21 cross terms is 2 × 10⁻¹⁶ of the MSE, and the seven terms sum to the MSE. |
| K. Season-mean split | Cross term 10⁻¹⁸ of the MSE. The season-mean part equals $\sum_s p_s (b_s - b)^2$ (74% of the seasonal term in this example). |
| L. Regional-mean series | The series MSE equals the mean bias plus the uniform seasonal and uniform anomaly parts, and the seasonal and anomaly terms each split orthogonally into uniform and local parts. |
| M. Area | Missed and excess errors have a cross term of 0, and their shares (0.578 and 0.380 of the reference area) sum to the total (0.959). |
| N. Driver-conditional split | Orthogonal (cross term 8 × 10⁻¹⁷), but it captures 100% of the mean bias term, 15% of the pattern term, 0% of the seasonal term, and 5% of the anomaly term. A sum of whole terms would capture each term either entirely or not at all. |

The checks are in [`experiments/verify_suite.py`](experiments/verify_suite.py).

## The volume adjustment

A component's share of the normalized MSE shrinks as the error is split into more components.
If the error were split into 1000 orthogonal components of similar size,
and each component were scored on its share,
every component would score near 1 even when the total scores poorly.
Scoring a component therefore needs a volume adjustment:
compare the component's error with the reference's variance in the same component,
not with the reference's total variance.
This section derives the adjustment and shows that it is the one that keeps component scores consistent with the total.
Checks O–R use the same synthetic data as [checks A–I](#the-orthogonal-decomposition-of-the-mse).

### The adjustment

Let $P_k$ be orthogonal projections onto fixed subspaces that decompose the error,
$e = \sum_k P_k e$,
as in the [orthogonal decomposition](#the-orthogonal-decomposition-of-the-mse).
Let $\tilde x$ be the reference with the benchmark removed that defines the normalizer,
so that $D = \lVert \tilde x \rVert^2$; for $D_{\mathrm{total}}$,
$\tilde x = x - \langle \bar x \rangle$.
The same projections split the reference's variance,
$D = \sum_k D_k$ with $D_k = \lVert P_k \tilde x \rVert^2$.
Define each component's volume and adjusted error as

$$v_k = \frac{D_k}{D}, \qquad E_k^2 = \frac{\lVert P_k e \rVert^2}{D_k}.$$

Then

$$\mathrm{NMSE} = \sum_k \frac{\lVert P_k e \rVert^2}{D} = \sum_k v_k E_k^2,$$

so the total is the volume-weighted mean of the adjusted component errors,
and each adjusted error is on the same scale as the total.
In the thought experiment, with 1000 components of equal volume and equal error,
every $E_k^2$ equals the NMSE, so every component scores the same as the total.
If all the error lies in one component,
that component's $E_k^2$ is 1000 times the NMSE,
and the other 999 components score 1 (check R).

The share $v_k E_k^2 / \mathrm{NMSE}$ says where the error is;
the adjusted score $s(E_k)$ says how well the model reproduces that component.
The scorecard should show both.
$1 - E_k^2$ is the mean squared skill score (Murphy, 1988) of component $k$ against a reference forecast with no variability in that component.

### Why this adjustment

- **Consistent at every level.** If component $k$ is split further into fixed subspaces with volumes $D_{kj}$,
  then $E_k^2 = \sum_j (D_{kj} / D_k) E_{kj}^2$.
  The adjusted error of any component is the volume-weighted mean of the adjusted errors of its parts,
  so splitting more finely never inflates the scores.
- **Already used for score maps.** Cells are disjoint,
  so they are orthogonal components.
  For the RMSE statistic, with $D_{\mathrm{time}}$,
  cell $i$ has volume $w_i \sigma_i^2 / \sum_j w_j \sigma_j^2$.
  Its adjusted error is its own normalized error, $\mathrm{crmse}_i / \sigma_i$,
  which is the value the RMSE score map scores.
  The regional error is then the volume-weighted mean;
  this is the weighting of each $(e_i / n_i)^2$ by $w_i n_i^2$ described on the [Scoring page](https://github.com/thodson-usgs/ilamb3/blob/5b4b2158ca1d3943e4a80f5fb89f22eec8a73075/docs/deep-dive/scoring.md?plain=1#L124-L128).
- **Measured by the signal,
  not by the number of dimensions.** An analysis of variance divides each sum of squares by its degrees of freedom,
  which measures volume by the number of dimensions.
  That assumes the reference's variance is spread evenly over the dimensions,
  and it is not.
  In check O,
  the seasonal subspace holds 11 of the 41 degrees of freedom in time in each cell,
  but 79% of the reference's variance.
  Dividing by the reference's variance compares each component's error with the signal that the model must reproduce in that component.

### Components with no reference volume

The reference has no variance about its benchmark in the subspace of constants,
so the mean bias has $D_0 = 0$.
With $D_{\mathrm{time}}$, the pattern term has zero volume as well.
These are bias terms,
where the reference has no variability for the model to reproduce.
Normalize them by $D$, as the Bias score and Murphy's (1988) bias term do.
With $D_{\mathrm{total}}$,

$$\mathrm{NMSE} = \frac{\langle b \rangle^2}{D} + v_{\mathrm{pat}} E_{\mathrm{pat}}^2 + v_{\mathrm{seas}} E_{\mathrm{seas}}^2 + v_{\mathrm{anom}} E_{\mathrm{anom}}^2,
\qquad v_{\mathrm{pat}} + v_{\mathrm{seas}} + v_{\mathrm{anom}} = 1.$$

Each adjusted error is the familiar normalized error for its concept:

- **Pattern.** $E_{\mathrm{pat}}^2 = \mathrm{Var}(b) / \sigma_r^2 = \hat\sigma^2 + 1 - 2 \hat\sigma R$,
  the normalized centered distance of the Taylor diagram,
  because $D_{\mathrm{pat}} = \sigma_r^2$.
- **Seasonal.** $E_{\mathrm{seas}}^2$ is the seasonal error relative to the variance of the reference's mean annual cycle.
- **Anomaly.** $E_{\mathrm{anom}}^2$ is the anomaly error relative to the variance of the reference's anomalies.

The shares and the adjusted errors give different pictures.
In check O, the anomaly term is 7% of the NMSE, the second smallest share,
but its adjusted error, 2.71, is the largest of the three,
because the reference's anomalies hold only 6.5% of its variance.
Shares understate how poorly a model reproduces a component that is small in the reference.
Components with a very small volume can have large and noisy adjusted errors,
as cells with almost no variability do in the score maps.
Their small volume keeps them from dominating the total.

### Score functions

Under $s = e^{-\alpha \cdot \mathrm{NMSE}}$ (Hodson et al., 2021, Eq. 15),
the adjustment has a product form:

$$s_{\mathrm{total}} = e^{-\alpha \langle b \rangle^2 / D} \prod_k \left( e^{-\alpha E_k^2} \right)^{v_k}.$$

The total score is the mean-bias score times the volume-weighted geometric mean of the adjusted component scores (check Q).
Under $e^{-E}$ there is no product form,
but the total is still the score of $\sqrt{\mathrm{NMSE}}$,
with the NMSE assembled as above, so the ranking remains invariant to $s$.

### Parts within a component

The amplitude, correlation,
and timing parts of the bisector splits are directions defined by the error itself,
not fixed subspaces, so the reference has no separate volume in each.
They share their parent's volume.
For the pattern, for example,

$$\frac{A_{\mathrm{pat}}}{D_{\mathrm{pat}}} = \tfrac12 (\hat\sigma - 1)^2 (1 + R),
\qquad \frac{C_{\mathrm{pat}}}{D_{\mathrm{pat}}} = \tfrac12 (\hat\sigma + 1)^2 (1 - R),$$

which sum to $E_{\mathrm{pat}}^2$;
the seasonal and anomaly parts work the same way.
Show these parts as a split of their parent's adjusted error rather than scoring each one:
scoring them separately would reintroduce the dilution,
since two equal parts would each carry half their parent's error.

### Scorecard

Each component then gets two numbers (placeholder values):

```
                    share of the total      adjusted score   volume
  Mean bias         ███████░░░░░░  xx%          0.xx           —
  Pattern           █░░░░░░░░░░░░  xx%          0.xx          xx%
  Seasonal          ███░░░░░░░░░░  xx%          0.xx          xx%
  Anomaly           ██░░░░░░░░░░░  xx%          0.xx          xx%
```

### Checks

All four checks pass.

| Check | Result |
|---|---|
| O. Volume identity | Volumes: pattern 14.6%, seasonal 78.8%, anomaly 6.5% of $D_{\mathrm{total}}$. Adjusted errors: pattern 0.370, seasonal 0.671, anomaly 2.712; mean bias 1.681. These reassemble the NMSE (2.4410), and the adjusted pattern error equals $\hat\sigma^2 + 1 - 2 \hat\sigma R$. |
| P. Many components | Total score 0.403. Split by cell (648 components), the median unadjusted score is 0.968 and the median adjusted score 0.392. Split by cell and calendar month (7776 components), the medians are 0.993 and 0.406. In both cases, the volume-weighted mean of the adjusted errors equals the total, and the per-cell adjusted error equals the RMSE score map's local error. |
| Q. Product form | With $\alpha = 1$: total score 0.0871 = mean-bias score 0.1863 × the volume-weighted geometric mean of the adjusted scores (pattern 0.6906, seasonal 0.5112, anomaly 0.0664). |
| R. 1000 equal volumes | With an NMSE of 0.5 spread evenly, each component's unadjusted score is 0.978 and its adjusted score is 0.493, the same as the total's. With all of the error in one component, that component's adjusted NMSE is 500 and the other 999 score 1. |

The checks are in [`experiments/verify_volume.py`](experiments/verify_volume.py).

## A general rule for volumes

The [volume adjustment](#the-volume-adjustment) set each component's volume to the reference's variance in that component.
That leaves a general question: what rule should assign volumes,
so that the same rule covers splits in space, time, frequency,
and anything else?
This section poses the problem, states two axioms,
derives the general solution from them,
and applies it to the cases that come up in ILAMB.
Checks S–W use a synthetic daily series.

### Problem

Let the error split orthogonally, $e = \sum_k P_k e$,
so that $\lVert e \rVert^2 = \sum_k \lVert P_k e \rVert^2$.
Give each component a volume $v_k \ge 0$ with $\sum_k v_k = 1$,
and score it on the adjusted error $E_k^2 = \lVert P_k e \rVert^2 / (v_k D)$.
Then $\mathrm{NMSE} = \sum_k v_k E_k^2$ for any choice of volumes (check S),
so the identity does not determine them.
Natural guesses are the duration fraction for time windows (1/4 for each of four equal seasons; 355/356 and 1/356 for a split into 355 days and 1 day),
the area fraction for regions, and 1/2 each for the bias and variance terms.
Should the volumes be these fractions or some function of them,
and is there a rule that gives volumes for frequency bands,
where one band may hold 99% of the spectrum and another 1%?

### Axioms

A component's share of the error is extensive:
it adds over components and grows with the component's size, as mass does.
A score needs an intensive quantity, one that does not depend on size,
as density divides mass by volume.
Two axioms determine the adjustment; everything else follows from them.

**Axiom A (conditional expectation).** Each component's adjusted error is its mean error per unit volume,
where the volumes are a single probability measure on components.
Equivalently,
for every orthogonal decomposition the law of total expectation holds,

$$\mathrm{NMSE} = \sum_k v(P_k)\, E_k^2, \qquad \sum_k v(P_k) = 1,$$

and the volume $v(P)$ of a component depends only on the component,
not on how the rest of the space is split.
In the law of total expectation, $\mathrm{E}[X] = \sum_k \Pr(A_k)\,
\mathrm{E}[X \mid A_k]$,
the volumes play the role of the probabilities and the adjusted errors the role of the conditional means.
Poverty measures with this property are called decomposable,
with population shares as the coefficients ([Foster, Greer, and Thorbecke, 1984](https://doi.org/10.2307/1913475)).
Within one decomposition the components commute,
and the volumes are an ordinary probability measure.
Across decompositions that do not commute,
such as time windows and frequency bands, they are a measure on subspaces,
the setting of Gleason's theorem below.

**Axiom B (independent of the model).** Volumes depend on the subspaces and the reference,
never on the model being scored.

### Consequences

1. **The adjustment factor is the reciprocal of the volume.** The mean error per unit volume in component $k$ is $E_k^2 = \lVert P_k e \rVert^2 / (v_k D)$.
   The law of total expectation alone gives the same factor for any adjusted error that is a multiple of its share,
   $E_k^2 = c_k \lVert P_k e \rVert^2 / D$:
   put all of the error in component $k$, and $v_k c_k = 1$ follows.
2. **Volumes are additive.** Let $P = Q + R$ with $Q$ and $R$ orthogonal.
   The decomposition $\{P, I - P\}$ gives $v(P) = 1 - v(I - P)$, and $\{Q, R,
   I - P\}$ gives $v(Q) + v(R) = 1 - v(I - P)$, so $v(P) = v(Q) + v(R)$.
3. **Consistency in aggregation.** A parent's adjusted error is the average of its parts' adjusted errors,
   with coefficients $v_j / v_P$ that sum to 1 by consequence 2,
   so computing the total in two stages gives the same result as computing it in one.
   This is the tower property of conditional expectation,
   of which the law of total expectation is the special case at the root.
   Index-number theory calls it consistency in aggregation (Vartia, 1976; [Diewert, 1978](https://doi.org/10.2307/1909755)).
   Splitting one component leaves every other component's volume and score unchanged (check T).
4. **Volumes are the measure fractions themselves,
   not a function of them.** If the volume is $f$ of a measure fraction,
   consequence 2 requires $f(a + b) = f(a) + f(b)$.
   With $f \ge 0$, the only solutions are linear (Cauchy's functional equation),
   and $f(1) = 1$ makes $f$ the identity.
5. **The general form.** Axiom A is the hypothesis of Gleason's theorem ([Gleason, 1957](https://doi.org/10.1512/iumj.1957.6.56050)):
   on a real or complex Hilbert space of dimension at least 3,
   every non-negative function on orthogonal projections that sums to 1 over every orthogonal decomposition of the identity has the form $v(P) = \mathrm{tr}(\rho P)$ for a positive semidefinite operator $\rho$ with trace 1. Since $\mathrm{tr}(\rho P) = \mathrm{E} \lVert P \varepsilon \rVert^2$ for a random field $\varepsilon$ with covariance $\rho$,
   it follows as a corollary that

$$v_k = \frac{\mathrm{E} \lVert P_k \varepsilon \rVert^2}{\mathrm{E} \lVert \varepsilon \rVert^2}.$$

**A component's volume is the share of a baseline error $\varepsilon$ that falls in it.** The axioms leave one choice,
the baseline's covariance, and Axiom B says it must come from the reference,
not the model.
The baseline defines what "error spread evenly" means:
if the model's error had the baseline's covariance,
the expected squared error in every component would be proportional to its volume,
so all components would score alike on average, for every decomposition.
Space, time, and frequency volumes are therefore not separate choices;
they are the diagonal of one covariance written in three bases.

### The two axioms are independent

- **Equal volumes per component**, $1/K$ for $K$ components,
  satisfy the law of total expectation in each decomposition,
  but a component's volume then depends on how the rest of the space is split,
  so they fail Axiom A. Splitting the variance term into two frequency bands drops the bias term's volume from 1/2 to 1/3 and changes its adjusted error from 0.062 to 0.092,
  although nothing about the bias changed (check T).
  The variance term's adjusted error also stops being an average of its bands' adjusted errors:
  the coefficients that reproduce it sum to 4/3 ([check AA](#volumes-and-adjustment-factors)).
- **The model's own error shares**,
  $v(P) = \lVert P e \rVert^2 / \lVert e \rVert^2$,
  are a probability measure on subspaces, so they satisfy Axiom A,
  but they fail Axiom B. Every component with any error then has an adjusted error equal to the NMSE,
  and the scorecard shows nothing.
  Axiom B is therefore not a consequence of Axiom A.

### Space, time, and frequency

**Time windows and regions.** For a projection onto a set of points,
the volume needs only the baseline's variance at each point, $\sigma_i^2$:

$$v_k = \frac{\sum_{i \in k} w_i \sigma_i^2}{\sum_i w_i \sigma_i^2},$$

where $w_i$ is the point's area or duration weight.
With equal variance everywhere (a white baseline, or any stationary one),
this is the duration or area fraction: 1/4 for each of four equal seasons,
and 355/356 and 1/356 for the day split,
with no other function applied (check U).
With the reference's own variance,
it is the duration or area fraction times the window's relative variance,
$p_k s_k^2 / \sum_j p_j s_j^2$,
where $s_k^2$ is the reference's mean square about the benchmark in window $k$.
This is the weighting the RMSE score maps already use for cells.

**Frequency bands.** The volume of a band is the baseline's power in the band,
its share of the baseline's spectrum.
It equals the bandwidth fraction only for a white baseline.
A stationary red baseline is homogeneous in time but not in frequency,
so the same $\rho$ gives time windows their duration fractions and gives low-frequency bands more than their bandwidth (checks U, V).
In the 99%/1% example,
the volumes are 0.99 and 0.01 if the shares are of the baseline's power;
if they are shares of bandwidth,
those are the volumes only for a white baseline.

**Bias and variance.** The mean is the zero-frequency component,
a single degree of freedom.
Its volume is $1/n$ under a white baseline,
$1/n_{\mathrm{eff}}$ under a stationary one, and 0 under the realized reference,
whose mean about the benchmark is zero by construction.
It would be 1/2 only under a baseline that puts half of its variance in the mean.

| Components | White baseline | Stationary AR(1) baseline, $\phi = 0.9$ | Realized reference |
|---|---|---|---|
| 355 days, 1 day | 355/356, 1/356 | 355/356, 1/356 | 0.99997, 0.00003 |
| Four equal seasons | 1/4 each | 1/4 each | 0.226, 0.281, 0.261, 0.232 |
| Mean, periods ≥ 19.8 days (10% of frequencies), the rest | 0.003, 0.101, 0.896 | 0.052, 0.746, 0.202 | 0, 0.967, 0.033 |
| Deciles of the reference | 0.1 each | 0.1 each | 0.252 and 0.250 for the outer deciles; 0.006 and 0.003 for the middle two |

### Choosing the baseline

I suggest the reference's own second moment about the benchmark,
$\rho \propto \mathrm{E}[\tilde x \tilde x^\top]$,
with the expectation taken over the reference's internal variability and its climatology held fixed.
The total's normalizer $D$ is the mean squared error of the benchmark (the reference's mean, used as a forecast),
so with this baseline each adjusted error compares the model's error in a component with the benchmark's error there.
With the realized reference,
$1 - E_k^2$ is exactly the component's mean squared skill score against the benchmark,
and the volumes are the ones the score maps use.

[The volume adjustment](#the-volume-adjustment) used the realized reference,
$\rho \propto \tilde x \tilde x^\top$,
which is the one-sample estimate of this $\rho$.
It is adequate for components that are fixed in advance and have many degrees of freedom,
such as the four terms of the tree and the cells of a score map.
It fails in two cases:

- **Small components are noisy.** A single day's realized volume is that day's squared anomaly:
  0.00003 instead of 1/356 in check U,
  because the reference happened to be near its mean that day.
  That would multiply the day's adjusted error by about 100.
- **Components selected by the reference's values are biased.** For deciles of the reference,
  the middle two deciles hold 0.3% and 0.6% of the realized volume instead of about 10%,
  because the reference is near its mean there by construction (check W).
  Hodson's quantile decomposition (Sec. 3.6) bins by the observations,
  so it needs the pooled estimate.

For fine splits and for splits selected by the reference's values,
estimate $\rho$ with pooling:
the realized climatology (the pattern and seasonal cycle are fixed structure),
plus a pooled covariance of the anomalies,
such as a variance for each cell and calendar month,
with autocorrelation if frequency bands are scored.
I have not tested a pooled estimator here.

Keep the treatment of the bias terms in [the volume adjustment](#the-volume-adjustment) (the mean bias, and the pattern under $D_{\mathrm{time}}$):
normalize them by $D$ and keep them outside the volume-weighted mean.
These are the subspaces where the benchmark is defined,
so the realized reference gives them no volume.
Including sampling variability would give them a volume,
the variance of the benchmark's estimate; for the mean,
that is $1/n_{\mathrm{eff}}$ in a stationary series (check V).
The adjusted mean-bias error would become $\langle b \rangle^2 / \mathrm{Var}(\bar x)$,
the squared bias in units of the standard error of the reference mean,
which is $n_{\mathrm{eff}}$ times the fallback's $\langle b \rangle^2 / D$.
That score measures whether a bias is detectable, not how large it is,
and it penalizes the same bias more as the reference record grows.

### Other decompositions

The same rule covers every orthogonal decomposition,
and the volume has the same form in each:

- **Partitions of points.** Regions, biomes, land and ocean, seasons, days,
  years.
  Partitions defined by the reference are included:
  quantile bins of the observations (Hodson, Sec. 3.6),
  the reference's extent and its complement (the Area analysis's missed and excess terms),
  and bins of an observed driver or regime ([check N](#components-and-diagnostics)).
  The volume is the baseline's variance summed over the points.
- **Averages.** The mean, the time mean of each cell, the climatology,
  and season means: the nested subspaces of the four-term tree.
  The volume is the baseline's variance in the averaged part.
- **Spectral.** Frequency, spatial wavenumber or spherical harmonics,
  and wavelet scales.
  The volume is the baseline's power in the band.
- **Bases adapted to the reference.** For EOFs of the baseline covariance,
  the volume of each EOF is its explained-variance fraction,
  because $\rho$ is diagonal in its own eigenbasis (check W).
- **Products.** Cell × calendar month, region × frequency band, and so on.
  A product of commuting projections is a projection,
  so the rule applies unchanged.
- **Variables.** Stacking variables needs an inner product that makes them commensurate.
  If variable $j$ is normalized by its own reference variance and given weight $c_j$,
  its realized volume is $c_j / \sum_i c_i$, its adjusted error is its own NMSE,
  and the total is the weighted mean of the variables' NMSEs.
  Equal weights give each of $J$ variables the volume $1/J$.
  The variable weights belong to the inner product, not to $\rho$.

Splits whose subspaces depend on the model fail Axiom B: the bisector, Murphy,
and quantile-coupling (scale, shape, sequence) splits.
They have no volume of their own,
and the scorecard shows them as splits of their parent's adjusted error,
as in [the volume adjustment](#the-volume-adjustment).

### Checks

Synthetic daily series of 356 days with an annual cycle and AR(1) variability ($\phi = 0.9$),
and a model with a bias, an error in the annual cycle, and its own noise.
Baselines: white, stationary AR(1), the realized reference,
and a random covariance.

| Check | Result |
|---|---|
| S. Identity | The NMSE (0.1549) is recovered for all 4 baselines × 4 decompositions (seasons, 355 + 1 days, frequency bands, a random orthogonal split). |
| T. Refinement | For every baseline, splitting the variance term into two bands leaves the mean's volume and adjusted error unchanged, and the variance term's adjusted error equals the volume-weighted mean of its parts'. With equal volumes per component, the same split changes the mean's adjusted error from 0.0615 to 0.0923. |
| U. Time windows | White and stationary baselines give 355/356 and 1/356 for the day split and 1/4 per season. The realized reference gives the day 0.00003 and the seasons 0.226, 0.281, 0.261, 0.232, equal to $p_k s_k^2 / \sum_j p_j s_j^2$. |
| V. Frequency bands | Mean, periods ≥ 19.8 days, rest: white 0.003, 0.101, 0.896; stationary 0.052, 0.746, 0.202; realized 0, 0.967, 0.033. The stationary mean volume equals $1/n_{\mathrm{eff}}$ ($n_{\mathrm{eff}} = 19.2$). |
| W. Quantile bins and EOFs | Decile volumes are the share of days in each decile (0.098–0.101) under the white and stationary baselines; under the realized reference, the outer deciles hold 0.252 and 0.250 and the middle two 0.006 and 0.003. EOF groups 1–5, 6–50, and the rest have volumes 0.053, 0.358, 0.589, equal to their explained-variance fractions. |

The checks are in [`experiments/verify_weights.py`](experiments/verify_weights.py).

## Volumes and adjustment factors

A component's share of the error understates how badly the model does in that component,
because the share holds only the fraction of the error that falls there.
Scored on its share,
every component scores better than the total (in [check P](#the-volume-adjustment), 648 cells have a median score of 0.968 against a total of 0.403).
The share is extensive: it grows with the size of the component.
The volume adjustment corrects this by dividing the share by the component's volume,
which gives an intensive error, the error per unit of volume.
With four seasons of equal length and an error spread evenly over the year,
each season holds 1/4 of the error.
Multiplying each season's share by 4 gives its adjusted error,
and the mean of the four adjusted errors is the total NMSE (checks X and Z).
In general,

$$E_k^2 = \frac{1}{v_k} \cdot \frac{\lVert P_k e \rVert^2}{D},
\qquad \sum_k v_k = 1, \qquad \mathrm{NMSE} = \sum_k v_k E_k^2.$$

$E_k^2$ is the NMSE the model would have if its error in every component were like its error in component $k$,
relative to the baseline.
If every component has the same adjusted error, that value is the total.

### Names

I suggest these names:

- **Volume**, $v_k$ (1/4 for a season).
  It is the component's share of the baseline's variance and its coefficient in the average that gives back the total,
  so the volumes sum to 1.
- **Adjustment factor**, $1/v_k$ (4 for a season).
  A component's share of the error is multiplied by this factor to give its adjusted error.
  The reciprocals of the factors sum to 1.
- **Extensive and intensive error.** The share, $\lVert P_k e \rVert^2 / D$,
  is the extensive error; the adjusted error, $E_k^2$, is the intensive error.

Calling 1/4 the weight matches its role as a coefficient of the average.
In survey sampling,
the sampling weight of a unit is the reciprocal of its inclusion probability,
which would make 4 the weight. ilamb3 already uses "weight" for cell areas,
time steps, and the weights of analyses in the overall score,
so I would avoid "weight" for either quantity here.

The factor is 4 when the baseline's variance is the same in every season.
With the reference-variance baseline recommended in the [previous section](#a-general-rule-for-volumes),
each season's factor is the reciprocal of its share of the reference's variance:
4.42, 3.56, 3.83, and 4.31 in check U.

### Bias and variance

In the average that gives back the MSE,
the coefficients of the bias and variance terms are their volumes,
and the factors are their reciprocals.
Axiom A in the [previous section](#a-general-rule-for-volumes) fixes them,
and both of its parts are needed.
The law of total expectation within this one decomposition holds for any coefficients that sum to 1,
including 1/2 and 1/2,
which give factors of 2 and 2. Equal coefficients as a rule are ruled out by the other part,
that a component's volume depends only on the component,
through its consequence, consistency in aggregation.
Split the variance term into two frequency bands and give every component an equal coefficient,
1/3:
the variance term's adjusted error is then no longer an average of its bands' adjusted errors,
because the coefficients that reproduce it sum to 4/3 (check AA).
With additive coefficients,
they sum to 1. A coefficient of 1/2 for the bias would require a baseline that puts half of its variance in the mean.

For an evenly spread error,
each coefficient is the share of that error that falls in the component,
and each factor is its reciprocal.
Equal seasons receive equal shares, so their coefficients are equal.
The mean is one degree of freedom out of $n$,
so an evenly spread error puts only $1/n$ of itself into the bias.
For $n = 356$, the coefficients are $1/356$ and $355/356$,
and the factors are 356 for the bias and $356/355 = 1.003$ for the variance (check Y).
With factors of 2 and 2,
a model whose error is spread evenly would get a bias error of 0.006 times the total and a variance error of 1.99 times the total.
With factors of 356 and 1.003, both equal the total.

With the reference-variance baseline,
the variance term holds all of the reference's variance, so its volume is 1,
its factor is 1, and its share is already its adjusted error.
The bias term has volume 0, so it has no finite factor.
It is reported as its share, $\langle b \rangle^2 / D$,
outside the weighted mean.
The scorecard entries are then $s(\langle b \rangle^2 / D)$ and $s(\mathrm{Var}(e) / D)$,
and the two arguments add up to the NMSE.

For the four-term tree in check O, the factors are 6.83 for the pattern,
1.27 for the seasonal term,
and 15.29 for the anomaly term (volumes of 14.6%, 78.8%, and 6.5%).
The mean bias has no factor and is reported as its share.

### Checks

Errors spread evenly (white noise) over 356 days, 20,000 draws.

| Check | Result |
|---|---|
| X. Seasons | Each season's mean share of the total is 0.250, 0.250, 0.250, and 0.249. After multiplying by 4, the ratios are 1.001, 1.000, 1.001, and 0.998. |
| Y. Bias and variance | Mean shares of the total are 0.0028 for the bias ($1/n$) and 0.9972 for the variance. Factors of 2 and 2 give 0.0056 and 1.994. Factors of 356 and 1.0028 give 1.003 and 1.000. |
| Z. Weighted mean | For a single draw, $\sum_k v_k E_k^2$ equals the total exactly, both for the seasons (volumes 1/4) and for bias and variance (volumes $1/n$ and $(n-1)/n$). |
| AA. Consistency in aggregation | The variance term is split into a low band (volume 0.1011) and a high band (0.8961). With additive volumes, the coefficients that make the variance term's adjusted error an average of its bands' are 0.1014 and 0.8986, which sum to 1. With equal volumes per component (1/2, then 1/3 each), they are 0.6667 and 0.6667, which sum to 4/3. |

The checks are in [`experiments/verify_factors.py`](experiments/verify_factors.py).

## The score function

With aggregate-then-score ([rubisco-sfa/ilamb3#217](https://github.com/rubisco-sfa/ilamb3/pull/217)),
the ranking of models does not depend on the score function,
because the error is aggregated before it is scored.
The score function still decides what the numbers mean and how the scorecard's component scores combine into the total. ilamb3 scores the relative RMSE,
$E = \sqrt{\mathrm{NMSE}}$, with $s = e^{-E}$ (`Collier2018`),
or with $\max(0, 1 - E)$ and a normalizer from regional quantiles (`RegionalQuantiles`).
This section argues for applying the exponential to the NMSE instead,
$s = e^{-\alpha\,\mathrm{NMSE}}$, gives its basis in probability theory,
and compares other summaries worth reporting: the percentage of the total,
a skill score, and an error interval.
Checks AB–AF reuse the synthetic data of check O.

### The exponential of the NMSE

**Independent evidence multiplies.** The NMSE adds over orthogonal components,
and the probabilities or likelihoods of independent events multiply.
The only continuous score with $s(0) = 1$ that turns sums into products,
$s(x + y) = s(x)\,s(y)$,
is $s(x) = e^{-\alpha x}$ (Cauchy's exponential functional equation).
The same argument makes $-\log p$ the only additive measure of information:
the NMSE plays the role of information, and the score the role of probability.
The current $e^{-\sqrt{x}}$ fails the product rule.
With $x = 0.7$ and $y = 1.9$,
$s(x + y) = 0.199$ but $s(x)\,s(y) = 0.109$ (check AB).

**The score is a likelihood.** Suppose the model's errors were independent and Gaussian,
with variance $c$ times the reference's variance.
Then the likelihood ratio of the model to a perfect model is $e^{-\alpha\,\mathrm{NMSE}}$,
with $\alpha = n / (2c)$ for $n$ independent data.
Orthogonal components of such errors are independent,
so the likelihood factorizes over components (check AC).
That factorization is the product form in [the volume adjustment](#the-volume-adjustment):
the total score is the mean-bias score times the volume-weighted geometric mean of the adjusted component scores (0.1842 for check O with $\alpha = \ln 2$, check AB).
When no likelihood is trusted,
[Bissiri, Holmes, and Walker (2016)](https://doi.org/10.1111/rssb.12158) derive $e^{-w \cdot \mathrm{loss}}$ as the coherent way to update beliefs with a loss function in place of a likelihood.
The Gaussian likelihood is their special case with squared error as the loss.

**The theory does not fix $\alpha$.** $\alpha = n / (2c)$ depends on an error variance that is not known in practice,
and Bissiri and others likewise leave the weight on the loss to be calibrated.
So $\alpha$ is a convention.
A natural one fixes the score of the benchmark,
the reference's mean used as a forecast, which has $\mathrm{NMSE} = 1$.
With $\alpha = \ln 2$, the benchmark scores 0.5,
a likelihood half that of a perfect model;
the current $\alpha = 1$ gives it $e^{-1} = 0.368$.
Every strictly decreasing score gives the same ranking,
so $\alpha$ changes only the displayed numbers.

**Averaging over $\alpha$.** By Bernstein's theorem,
every completely monotone score is a mixture of exponential scores over $\alpha$.
Averaging $e^{-\alpha\,\mathrm{NMSE}}$ over $\alpha \sim \mathrm{Exp}(1)$ gives $1 / (1 + \mathrm{NMSE})$,
the normalized Nash–Sutcliffe efficiency used in hydrology,
$1 / (2 - \mathrm{NSE})$.
Averaging over a gamma distribution gives $(1 + p\,\mathrm{NMSE})^{-1/p}$ (check AD).
Mixtures lose the product rule:
their totals are harmonic or power means of the component scores.

### How the total relates to the component scores

| Score | Total from the component scores | Range |
|---|---|---|
| $1 - \mathrm{NMSE}$ (MSE skill score) | Volume-weighted arithmetic mean, minus the mean-bias share | $(-\infty, 1]$ |
| $e^{-\alpha\,\mathrm{NMSE}}$ | Mean-bias score × volume-weighted geometric mean | $(0, 1]$ |
| $1 / (1 + \mathrm{NMSE})$ | Volume-weighted harmonic mean, for components without mean bias | $(0, 1]$ |
| $e^{-\sqrt{\mathrm{NMSE}}}$ (current) | No closed form | $(0, 1]$ |

No bounded score can make the total the arithmetic mean of the component scores.
That would require $s\left(\sum_k v_k x_k\right) = \sum_k v_k s(x_k)$ for all volumes and errors,
so $s$ would be affine in the NMSE,
and a decreasing affine function is unbounded below.
Among bounded scores,
the exponential is the only one whose total is a product of component scores.

### Other summaries for the scorecard

These do not replace the total score,
which should stay a single strictly decreasing function of the NMSE.
They are other ways to report the same decomposition.

**Percentage of the total.** Each component's share of the MSE, as a percentage,
says where the error is.
Because the components are orthogonal, the shares add to 100%,
and the component errors in the variable's units add in quadrature to the RMSE.
For the check O model, the RMSE of 3.587 splits into a mean bias of 2.977,
pattern 0.535, seasonal 1.670, and anomaly 0.967, which are 68.8%, 2.2%, 21.7%,
and 7.3% of the MSE (check AF).
A share is extensive: it shows where the error is,
not how well the model reproduces a component.
The adjusted score shows that.

**Skill score as a percentage.** $100\,(1 - \mathrm{NMSE})$ is the percentage of the benchmark's MSE that the model removes:
the MSE skill score, or the Nash–Sutcliffe efficiency in hydrology.
It is the most direct reading of the NMSE,
and its total is the volume-weighted mean of the component skill scores,
minus the mean-bias share.
It is negative for a model worse than the benchmark:
−144% for the check O model.

**Error interval.** Reporting the error as $b \pm 2 s_e$ in the variable's units,
where $b$ is the mean error and $s_e$ its standard deviation,
gives a range a reader can use directly.
This is standard practice in metrology.
NIST reports an expanded uncertainty $U = k\,u_c$ with coverage factor $k = 2$ by convention,
an interval with a level of confidence of about 95% when the normal distribution applies ([Taylor and Kuyatt, 1994, Sec. 6](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-6-expanded-uncertainty)).
The interval does not need normal errors to be meaningful,
because its coverage has distribution-free bounds:

- At least 3/4 of any distribution lies within the mean ± 2 standard deviations (Chebyshev's inequality).
  The bound is attained by a three-point distribution.
- At least 8/9 of any unimodal distribution does (the Vysochanskij–Petunin inequality).
  [Pukelsheim (1994)](https://doi.org/10.1080/00031305.1994.10476030) used the same inequality to justify the three-sigma rule:
  at least 95% of any unimodal distribution lies within three standard deviations of the mean.
- 95.4% of a normal distribution does.
  For six unimodal distributions, including skewed and heavy-tailed ones,
  the coverage is 0.941 to 1.000 (check AE).

Two cautions apply.
First, the interval describes the spread of the model's errors over the domain;
it is not a prediction interval for a new value,
which would need the model's predictive distribution and the sampling uncertainty of the estimates,
and the errors are correlated in space and time.
The bounds above do hold exactly for the area- and time-weighted distribution of the errors,
because that is itself a distribution: for the check O model,
95.6% of the domain has errors within $b \pm 2 s_e$.
Second, report $b \pm 2 s_e$ rather than $\pm 2\,\mathrm{RMSE}$.
Both contain at least 3/4 of the errors (for $\pm 2\,\mathrm{RMSE}$, by Markov's inequality applied to the squared error),
but $\pm 2\,\mathrm{RMSE}$ is symmetric about zero and hides the sign of the bias.
For the check O model, $b \pm 2 s_e$ is $[-6.98, 1.03]$,
while $\pm 2\,\mathrm{RMSE}$ is $[-7.17, 7.17]$, 1.8 times as wide.

### Recommendation

1. Score the total with $e^{-\alpha\,\mathrm{NMSE}}$ and $\alpha = \ln 2$,
   so the benchmark scores 0.5, in place of $e^{-E}$.
   Rankings do not change.
2. In the scorecard,
   show each component's percentage of the total MSE and its error in the variable's units next to its adjusted score.
3. Report the total error as $b \pm 2 s_e$ in the variable's units,
   described as the spread of the errors over the domain rather than as a prediction interval.

### Checks

| Check | Result |
|---|---|
| AB. Product rule | With $x = 0.7$, $y = 1.9$: $e^{-x \ln 2}$ gives $s(x+y) = s(x)\,s(y) = 0.165$; $e^{-\sqrt{x}}$ gives 0.199 and 0.109; $1/(1+x)$ gives 0.278 and 0.203. For check O, the total equals the mean-bias score times the volume-weighted geometric mean of the adjusted scores: 0.1842 with $\alpha = \ln 2$ and 0.0871 with $\alpha = 1$. |
| AC. Gaussian likelihood | For 500 Gaussian errors with $c = 250$, the likelihood ratio to a perfect model, 0.7199, equals $e^{-(n/2c)\,\mathrm{NMSE}}$. The log-likelihood is the sum over two orthogonal components, whose squared norms are uncorrelated over 20,000 isotropic draws (−0.001). |
| AD. Mixtures over $\alpha$ | Exponential and gamma mixtures of $e^{-\alpha x}$ equal $1/(1+x)$ and $(1 + x/2)^{-2}$ at $x$ = 0.25, 1, and 4. For five random components, the total $1/(1 + \sum_k v_k E_k^2)$ equals the volume-weighted harmonic mean of the component scores (0.5297). |
| AE. Error intervals | Coverage of the mean ± 2 standard deviations: normal 0.954, Laplace 0.941, gamma(2) 0.953, lognormal 0.963, Student $t_3$ 0.959, uniform 1.000; Chebyshev's three-point case 0.750. Check O model errors (area and time weighted): $b = -2.977$, $s_e = 2.002$, RMSE 3.587; 0.956 of the domain lies within $b \pm 2 s_e$ and 0.981 within $\pm 2\,\mathrm{RMSE}$. |
| AF. Error budget | The RMSE that `scoring.aggregate` pools over time and area, 3.587, is the square root of the sum of squares of the mean bias 2.977, pattern 0.535, seasonal 1.670, and anomaly 0.967, which are 68.8%, 2.2%, 21.7%, and 7.3% of the MSE. Skill score $1 - \mathrm{NMSE} = -1.441$. |

The checks are in [`experiments/verify_scores.py`](experiments/verify_scores.py).

## Verification

Every number in this document comes from a script in [`experiments/`](experiments/).
Run the scripts from an `ilamb3` checkout with the development environment installed.
Each prints one line per check, marked `PASS` or `FAIL`.
All 45 checks pass on the `fix/aggregate-then-score` branch.

| Script | Checks | Section | Depends on |
|---|---|---|---|
| [`verify_fixes.py`](experiments/verify_fixes.py) | 1–12, 14 | [Aggregating errors across components](#aggregating-errors-across-components) | `verify_components.py`, `ilamb3` |
| [`verify_components.py`](experiments/verify_components.py) | A–I | [The orthogonal decomposition of the MSE](#the-orthogonal-decomposition-of-the-mse) | `ilamb3` |
| [`verify_suite.py`](experiments/verify_suite.py) | J–N | [Components and diagnostics](#components-and-diagnostics) | `verify_components.py` |
| [`verify_volume.py`](experiments/verify_volume.py) | O–R | [The volume adjustment](#the-volume-adjustment) | `verify_components.py` |
| [`verify_weights.py`](experiments/verify_weights.py) | S–W | [A general rule for volumes](#a-general-rule-for-volumes) | NumPy |
| [`verify_factors.py`](experiments/verify_factors.py) | X–Z, AA | [Volumes and adjustment factors](#volumes-and-adjustment-factors) | NumPy |
| [`verify_scores.py`](experiments/verify_scores.py) | AB–AF | [The score function](#the-score-function) | `verify_volume.py`, `ilamb3`, SciPy |

```bash
for f in docs/design/experiments/verify_*.py; do python "$f"; done
```

## References

- Bissiri, P. G., Holmes, C. C., & Walker, S. G. (2016).
  A general framework for updating belief distributions. *Journal of the Royal Statistical Society Series B:
  Statistical Methodology*, 78(5), 1103–1130. https://doi.org/10.1111/rssb.12158
- Collier, N., Hoffman, F. M., Lawrence, D. M., Keppel-Aleks, G., Koven, C. D.,
  Riley, W. J., Mu, M., & Randerson, J. T. (2018).
  The International Land Model Benchmarking (ILAMB) system: Design, theory,
  and implementation. *Journal of Advances in Modeling Earth Systems*, 10(11),
  2731–2754. https://doi.org/10.1029/2018MS001354
- Cox, R. T. (1946).
  Probability,
  frequency and reasonable expectation. *American Journal of Physics*, 14(1),
  1–13. https://doi.org/10.1119/1.1990764
- Diewert, W. E. (1978).
  Superlative index numbers and consistency in aggregation. *Econometrica*,
  46(4), 883–900. https://doi.org/10.2307/1909755
- Foster, J., Greer, J., & Thorbecke, E. (1984).
  A class of decomposable poverty measures. *Econometrica*, 52(3),
  761–766. https://doi.org/10.2307/1913475
- Gleason, A. M. (1957).
  Measures on the closed subspaces of a Hilbert space. *Indiana University Mathematics Journal*,
  6(4), 885–893. https://doi.org/10.1512/iumj.1957.6.56050
- Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009).
  Decomposition of the mean squared error and NSE performance criteria:
  Implications for improving hydrological modelling. *Journal of Hydrology*,
  377(1–2), 80–91. https://doi.org/10.1016/j.jhydrol.2009.08.003
- Hodson, T. O., Over, T. M., & Foks, S. S. (2021).
  Mean squared error,
  deconstructed. *Journal of Advances in Modeling Earth Systems*, 13,
  e2021MS002681. https://doi.org/10.1029/2021MS002681
- Jaynes, E. T. (2003). *Probability Theory: The Logic of Science*.
  Cambridge University Press. https://doi.org/10.1017/CBO9780511790423
- Murphy, A. H. (1988).
  Skill scores based on the mean square error and their relationships to the correlation coefficient. *Monthly Weather Review*,
  116(12),
  2417–2424. https://doi.org/10.1175/1520-0493(1988)116<2417:SSBOTM>2.0.CO;2
- Pukelsheim, F. (1994).
  The three sigma rule. *The American Statistician*, 48(2),
  88–91. https://doi.org/10.1080/00031305.1994.10476030
- Taylor, B. N., & Kuyatt,
  C. E. (1994). *Guidelines for evaluating and expressing the uncertainty of NIST measurement results* (NIST Technical Note 1297).
  National Institute of Standards and Technology. https://www.nist.gov/pml/nist-technical-note-1297
- Taylor, K. E. (2001).
  Summarizing multiple aspects of model performance in a single diagram. *Journal of Geophysical Research:
  Atmospheres*, 106, 7183–7192. https://doi.org/10.1029/2000JD900719
- Vartia, Y. O. (1976).
  Ideal log-change index numbers. *Scandinavian Journal of Statistics*, 3(3),
  121–126.
- World Meteorological Organization (2007).
  Standardised Verification System (SVS) for Long-Range Forecasts (LRF). *Manual on the Global Data-processing and Forecasting System* (WMO-No. 485),
  Attachment II.8. https://www.metoffice.gov.uk/binaries/content/assets/metofficegovuk/pdf/research/climate-science/climate-observations-projections-and-impacts/svslrf.pdf
- ILAMB v2 source at commit 8e58622:
  [score weights](https://github.com/rubisco-sfa/ILAMB/blob/8e58622c5a107327d7237554869c2c4e63633616/src/ILAMB/Confrontation.py#L292-L300),
  [`computeOverallScore`](https://github.com/rubisco-sfa/ILAMB/blob/8e58622c5a107327d7237554869c2c4e63633616/src/ILAMB/Confrontation.py#L671-L711),
  and [the scoreboard roll-up](https://github.com/rubisco-sfa/ILAMB/blob/8e58622c5a107327d7237554869c2c4e63633616/src/ILAMB/Scoreboard.py#L340-L352).
