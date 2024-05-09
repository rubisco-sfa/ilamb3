# Preliminary Definitions

Some of the material here is adapted from [Collier, et al., 2018](https://doi.org/10.1029/2018MS001354). Please consult this publication for more thorough information and discussion.

In these Methods documentation pages, we explain the analysis of a generalized variable {math}`v(t,\mathbf{x})`, which we assume represents a piecewise discontinuous function of constants in space and time.

- The temporal domain is represented by the variable {math}`t` and is defined by the beginning and ending of time intervals.
- The spatial domain is represented by the variable {math}`\mathbf{x}` (in bold to emphasize it is a vector quantity), and is defined by the areas created by cell boundaries or those associated with data sites.
- A subscript {math}`v_{\mathrm{ref}}` reflects that the variable is treated as the reference, usually an observational product.
- A subscript {math}`v_{\mathrm{com}}` reflects that the variable is treated as the comparison, usually a modeled quantity.

## Mean Values Over Time

When calculating mean values over the time period, denoted by a bar superscribing the variable, we use the midpoint quadrature rule to approximate the integral,

```{math}
\begin{align*}
\overline{v}(\mathbf{x}) &= \frac{1}{t_f-t_i}\int_{t_i}^{t_f} v(t,\mathbf{x})\ dt\\
&\approx \frac{1}{T(\mathbf{x})} \sum_i^n v(t_i,\mathbf{x}) \Delta t_i
\end{align*}
```

where {math}`n` represents the number of time intervals on which {math}`v` is defined between the initial time {math}`t_i` and the final time {math}`t_f`, and {math}`\Delta t_i` is the size of the {math}`i^{\mathrm{th}}` time interval, modified to exclude time, which falls outside of the integral limits. The average value is obtained by dividing through by the amount of time in the interval, tf−t0, replaced in our discrete approximation by the following function.

```{math}
T(\mathbf{x}) = \sum_i^n \Delta t_i\ \mathrm{if}\ v(t_i,\mathbf{x})\ \text{is valid}
```

That is to say, if a variable has some values marked as invalid at some locations, we do not penalize the averaged value by including this as a time at which a value is expected. If an integral is desired instead of an average, then we simply omit the division by {math}`T(\mathbf{x})`.

## Mean Values Over Space

When computing spatial means over various regions of interest, denoted by a double bar over a variable, we use the midpoint rule for integration to approximate the following weighted spatial integral,

```{math}
\begin{align*}
\overline{\overline{v}}(t) &= \frac{1}{\int_\Omega w(\mathbf{x}) d\Omega}\int_\Omega v(t,\mathbf{x})w(\mathbf{x})\ d\Omega\\
&\approx \frac{1}{A(\Omega)} \sum_i^{n(\Omega)} v(t,\mathbf{x_i}) w(\mathbf{x}_i) a_i
\end{align*}
```

over a region {math}`\Omega`, also referred to as a area-weighted mean. Here the function {math}`w(\mathbf{x})` is an optional generic weighting function defined over space. The summation is over {math}`n(\Omega)`, that is, the integer number of spatial cells whose centroids fall into the region of interest. A function evaluation at a location {math}`\mathbf{x}_i` refers to the constant value which corresponds to that spatial cell. The value of {math}`a_i` is the area of the cell, which could be some fraction of the total cell area if integrating over land in coastal regions. We then divide through by the measure, the sum of the grid areas with the weights,

```{math}
A(\Omega) = \sum_i^{n(\Omega)} w(\mathbf{x}_i) a_i\ \text{if}\ v(t,\mathbf{x_i})\ \text{is valid}
```

If an integral only is required, we simply omit the division by {math}`A(\Omega)`. In cases where a mean over a collection of sites is needed, the spatial integral reduces to an arithmetic mean across the sites.

## Nested Grids

If we are spatially integrating a variable from a single source, then its spatial grid is clearly defined. However, if the integrand involves quantities from two different sources (like a bias or RMSE), then there is likely a disparity in both resolution and representation of land areas in the spatial grids. We address resolution differences by interpolating both sources to a grid composed of the cell *breaks* (the location at which two neighboring cells meet) of both data sources.

For example, consider the cell breaks of two coarse grids {math}`\mathcal{G}_1` and {math}`\mathcal{G}_2` below. We form a new grid composed of the unique values {math}`\mathcal{G}_c` and interpolate quantities to this new grid by nearest neighbor interpolation.

```{math}
\begin{align*}
\mathcal{G}_1 &= \{-90,-45,0,45,90\}\\
\mathcal{G}_2 &= \{-90,-30,30,90\}\\
\mathcal{G}_c &= \{-90,-45,-30,0,30,45,90\}
\end{align*}
```

Figure (a) below demonstrates this process. The cyan curve represents a step function defined on {math}`\mathcal{G}_1` and the magenta curve on {math}`\mathcal{G}_2`. Both are interpolated to the composed grid {math}`\mathcal{G}_c` without loss of information, albeit on a new grid containing more cells of variable size. Once on the composed grid, the quantities may be compared directly.

[<img src=https://agupubs.onlinelibrary.wiley.com/cms/asset/e8728828-9ec9-4738-a50c-35064e59310a/jame20779-fig-0002-m.jpg>](https://doi.org/10.1029/2018MS001354)

## Land Representation

While composing grids handles the spatial resolution differences without error, there are also differences in the representation of land that are challenging to resolve. Consider figure (b) above where we plot a section of the Caribbean showing differences in land representation of a fine scale grid {math}`\mathcal{L}_1` and a coarse grid {math}`\mathcal{L}_2`. The figure then highlights various different combinations of these representations. We consider this problem is not something we can resolve and therefore when bias or other comparisons are reported in ILAMB, they come from the composed grids where both sources report land was present (the red in figure (b)). This means that if you are comparing the mean values (reported on their original grids and land representations) to the bias, you may see what seems like an inconsistency.
