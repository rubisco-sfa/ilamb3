# Bias

This analysis computes the bias as the difference in time mean of the comparison variable with respect to a reference.

```{math}
b(\mathbf{x}) =
\overline{v}_{\mathrm{com}}(\mathbf{x}) -
\overline{v}_{\mathrm{ref}}(\mathbf{x})
```

Some reference data products now contain estimates of uncertainty, which we will designate here as {math}`\delta(t,\mathbf{x})`. We will include uncertainty in the exposition below with the understanding that {math}`\delta(t,\mathbf{x})=0` if not present in the reference dataset.

The bias methodology implementation uses several keyword options which you can use to control the method used to develop a score.

## Collier2018

To use this scoring method, pass the option `method="Collier2018"`. If the reference data is not temporal or represents only a single time slice, we use the traditional definition of relative error on the absolute value of the bias,

```{math}
\varepsilon(\mathbf{x}) =
\frac{
\left|b(\mathbf{x})\right|
}{\overline{v}_{\mathrm{ref}}(\mathbf{x})}
```

This is not ideal, nor unit invariant, but is all we can do with so little information. If the reference data spans a longer temporal record, we take the relative error to be,

```{math}
\varepsilon(\mathbf{x}) =
\frac{
\left|b(\mathbf{x})\right|
}{\mathrm{std}(v_{\mathrm{ref}}(t,\mathbf{x}))}
```

where {math}`\mathrm{std}` is the standard deviation operator in the temporal dimension. This implicitly normalizes the magnitude of errors by the amount of variability in any given year. We then use the exponential function to create a score map,

```{math}
s(\mathbf{x}) = e^{-\varepsilon(\mathbf{x})}
```

and then a scalar score by taking spatial means,

```{math}
S = \overline{\overline{s}}(\mathbf{x})
```

### Mass Weighting

The use of the standard deviation to normalize errors leads to the consequence that in areas where the given variable {math}`v` has a small magnitude, simple noise can lead to large relative errors. This happens in many land variables over hot and arid regions where little vegetation is encountered. Given the small contribution, it is undesirable that these errors induce a large negative contribution to the overall score. To address this issue, we introduce the concept of mass weighting. That is, when performing the spatial integral to obtain a scalar score {math}`S`, we weight the integral of {math}`s(\mathbf{x})` with the temporal mean of the reference, {math}`w(\mathbf{x}) = \overline{v}_{\mathrm{ref}}(\mathbf{x})`. Pass the option `mass_weighting=True` to utilize mass weighting (disabled by default).

## Regional Quantiles

To use this scoring method, pass the option `method="RegionalQuantiles"`. We developed this scoring method as an alternative to Collier2018 because we observe as a consequence to methodological choices, a strong correlation of regional tropics scores with the overall scores. In other words, performance in the tropics tends to be the most important, marginalizing gains made elsewhere.

The main idea behind this method is an appreciation that in Collier2018, normalizing by the standard deviation of the reference dataset is an attempt to make errors commensurate in order of magnitude across the globe. We suggest accomplishing the same goal by using *regional quantiles*. We begin with a set of global regions which correspond to some sense of biomes. We argue that the errors found within any given biome should be considered on the same order of magnitude.

For each variable and biome we compute quantiles of the absolute value of the bias across all datasets and a range of models from the CMIP5 and CMIP6 eras. These quantiles represent a distribution of model performance over the last two generations of Earth system modeling and contextualize what should be considered a large error. To browse our database of quantiles based on [Whittaker biomes](https://en.wikipedia.org/wiki/Biome#Whittaker_(1962,_1970,_1975)_biome-types), download the pandas dataframe stored [here](https://github.com/rubisco-sfa/ILAMB/raw/master/src/ILAMB/data/quantiles_Whittaker_cmip5v6.parquet).

Once developed, we can use the quantiles to score biases. If we use {math}`q_{70}(\mathbf{x})` to represent an extension of the 70th quantile to the globe, then the relative error can be defined as,

```{math}
\varepsilon(\mathbf{x}) = \frac{\left|b(\mathbf{x})\right|}{q_{70}(\mathbf{x})}
```

and then the score,

```{math}
s(\mathbf{x}) = \max(1 -  \varepsilon(\mathbf{x}),0)
```

The scalar score can then be found by simple spatial integral avoiding the need for mass weighting. Note that if you want to use this method, you must supply a `quantile_dbase` which is a pandas dataframe with the quantile information and possibly a `quantile_threshold` if you wish to use a value other than 70.

## Uncertainty

We have expanded the methodology to account for reference data uncertainty when present. Our current paradigm is to use the uncertainty to discount biases that fall within the window. Another way to think of this is that we only count biases which surpass the uncertainty, leading to scores which are better indicators of large and certain errors.

To accomplish this, we replace the absolute value of the bias in the relative error with the following:

```{math}
\left|b(\mathbf{x})\right| \rightarrow \max(\left|b(\mathbf{x})\right| - \overline{\delta}(\mathbf{x}),0)
```

Note that this option is enabled by default and works for both implemented scoring methods.
