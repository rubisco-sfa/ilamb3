# Global Net Ecosystem Carbon Balance

As the reference data products for net ecosystem carbon balance represent global totals, the general ILAMB methodology is not useful in gauging model performance. The following details the ILAMB method for comparing reference products ([Global Carbon Project](http://www.earth-syst-sci-data.net/8/605/2016/) and [Hoffman, et al. 2014](https://doi.org/10.1002/2013JG002381)) to model output.

This method expects the model variable `nbp` but we also accept `netAtmosLandCO2Flux`. The scores developed below are based on the global {math}`\mathit{nbp}`, accumulated starting from the first year of the reference product, represented as a generic {math}`v` below for ease of exposition. The reference products provide a two-sided estimate of uncertainty. We will designate {math}`v_{\mathrm{ref}}^{\mathrm{low}}(t)` to correspond to the lower values and
{math}`v_{\mathrm{ref}}^{\mathrm{high}}(t)` to the higher values. When a scalar uncertainty is required, we use the harmonic mean,

```{math}
\delta(t) = \sqrt{
    \left(v_{\mathrm{ref}}(t)-v_{\mathrm{ref}}^{\mathrm{low}}(t)\right)^2 +
    \left(v_{\mathrm{ref}}^{\mathrm{high}}(t)-v_{\mathrm{ref}}(t)\right)^2
}
```

## Difference Score

We assess a model's balance at the end of the reference data (or specified using the option `evaluation_year`) represented as {math}`y_e` in the equations below. We compute a relative error in the accumulated {math}`nbp` by normalizing by the uncertainty,

```{math}
\varepsilon =
\frac{
v_{\mathrm{com}}(y_e) -
v_{\mathrm{ref}}(y_e)
}{\delta(y_e)}
```

and then score using,

```{math}
S_{\mathrm{diff}} = e^{-\alpha |\varepsilon|}
```

where {math}`\alpha = -\ln(0.5) / 1`, chosen such that model differences outside of the uncertainty window will score less than a 0.5.

## Trajectory Score

While the balance at the end of the reference data is of primary concern, the difference score can favor models whose accumulated {math}`nbp` serendipitously matches at the end of the time period due to possibly large canceling positive and negative errors. To address this consequence, we also score the accumulated trajectory. We compute a relative error as a function of time where we only penalize the error that goes beyond the uncertainty window and also is normalized by its magnitude,

```{math}
\varepsilon(t) = \left(
    \max(v_{\mathrm{com}}(t)-v_{\mathrm{ref}}^{\mathrm{low}}(t),0) +
    \max(v_{\mathrm{ref}}^{\mathrm{high}}(t)-v_{\mathrm{com}}(t),0)
    \right) / \delta(t).
```

Then the score can be computed in the usual way,

```{math}
\begin{align*}
s(t) &= e^{-\varepsilon}\\
S_{\mathrm{traj}} &= \overline{s(t)}.
\end{align*}
```
