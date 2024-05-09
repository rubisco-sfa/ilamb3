# Bias

This analysis computes the bias as the difference in time mean of the comparison variable with respect to a reference.

```{math}
b(\mathbf{x}) =
\overline{v}_{\mathrm{com}}(\mathbf{x}) -
\overline{v}_{\mathrm{ref}}(\mathbf{x})
```

Some reference data products now contain estimates of uncertainty, which we will designate here as {math}`\delta(t,\mathbf{x})`. We will include uncertainty in the exposition below with the understanding that {math}`\delta(t,\mathbf{x})=0` if not present in the reference dataset.

The bias methodology implementation `ilamb3.analysis.bias_analysis` uses several keyword options which you can use to control

## Collier2018

If the reference data

```{math}
\varepsilon(\mathbf{x}) =
\frac{
\left|b(\mathbf{x})\right| -
\overline{\delta}(\mathbf{x})
}{\mathrm{std}(v_{\mathrm{ref}}(t,\mathbf{x}))}
```

## Regional Quantiles


blah
