# Relationships

As models are frequently calibrated using the mean state measures, a higher score does not necessarily reflect a more process-oriented model. In order to assess the representation of processes in models, we also evaluate so-called variable-to-variable relationships.

For the purposes of this section, we represent a generic dependent variable as {math}`v` and score its relationship with an independent variable {math}`u`. We form 2 discrete approximations (histograms) using the data collocated in the same spatiotemporal grid cells. The first is a 2D distribution which can be thought as a fraction of the datapoints found in each bin of the independent and dependent variables,

```{math}
\mathcal{F}_{\mathit{2D}}\left(v,u\right)
```

We make plots of these functions similar to panels (b) and (c) in the figure below. These example plots come from [Collier, et al., 2018](https://doi.org/10.1029/2018MS001354) and represent the relationship between gross primary productivity and surface air temperature. Panel (b) represents a reference product relationship and panel (c) a model relationship.

[<img src=https://agupubs.onlinelibrary.wiley.com/cms/asset/db948555-3ca8-4a2c-8b4a-622de8109d47/jame20779-fig-0009-m.jpg>](https://doi.org/10.1029/2018MS001354)

We also construct a discrete approximation of the independent variable {math}`u` in bins of the dependent variable {math}`v`, represented as

```{math}
\mathcal{U}\left(v\right)
```

In the plots above, the black curve represents the reference response and the maroon the comparison. In the scalar output, we report a score based on the relative RMSE between responses,

```{math}
\begin{align*}
\varepsilon &= \frac{
\sqrt{
\int \left(\mathcal{U}_{\mathrm{com}}\left(v\right) - \mathcal{U}_{\mathrm{ref}}\left(v\right)\right)^2\ dv
}
}{
 \int \mathcal{U}_{\mathrm{ref}}\left(v\right)^2\ dv
}\\
S &= e^{-\varepsilon}
\end{align*}
```
