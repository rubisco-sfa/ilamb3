---
status: proposed
---

# Score a component on its error per unit volume

A component's share of the NMSE grows with the size of its subspace,
so scores of the shares rank components by size as well as by error.
We propose to score each component on its adjusted error,
its share divided by its volume,
because conditional expectation fixes this adjustment,
and it keeps component scores consistent with the total.

## Context

[ADR 0005](0005-components-and-diagnostics.md) admits only orthogonal components of the MSE into the total,
$e = \sum_k P_k e$,
so that the shares $\lVert P_k e \rVert^2 / D$ add up to the NMSE.
A share is extensive.
In [check P](../design/probabilistic-framework.md#the-volume-adjustment),
a comparison with a total score of 0.403 is split into 648 cells,
and the median score of the cells' shares is 0.968, far above the total.
If the error were split into 1000 components with equal error,
every component would score near 1, however poorly the total scores.
No code on this branch computes volumes or adjusted errors.

## Decision

Give each component a volume $v_k \ge 0$,
computed from a baseline covariance $\rho$, and score it on its adjusted error:

$$v_k = \frac{\mathrm{tr}(\rho P_k)}{\mathrm{tr}(\rho)},
\qquad E_k^2 = \frac{1}{v_k} \cdot \frac{\lVert P_k e \rVert^2}{D},
\qquad \sum_k v_k = 1, \qquad \mathrm{NMSE} = \sum_k v_k E_k^2.$$

- The component score is $s(E_k)$ for the score function $s$ in use.
  The scorecard shows both the share, which says where the error is,
  and the adjusted score,
  which says how well the comparison reproduces that component.
- A bias term,
  a component in which the reference has no variance about the benchmark,
  has zero volume under the realized reference.
  The bias terms are the mean bias and the pattern under $D_{\mathrm{time}}$.
  Each is reported as its share, such as $\langle b \rangle^2 / D$,
  outside the volume-weighted mean,
  as in [components with no reference volume](../design/probabilistic-framework.md#components-with-no-reference-volume).
- Parts, such as the amplitude and correlation parts of a component,
  are directions defined by the error itself.
  They share their component's volume,
  are shown as a split of its adjusted error, and are never scored separately,
  as in [parts within a component](../design/probabilistic-framework.md#parts-within-a-component).
- The proposed baseline is the reference's second moment about the benchmark,
  $\rho \propto \mathrm{E}[\tilde x \tilde x^\top]$.
  The realized reference estimates it,
  for fixed components with many degrees of freedom.
  A pooled estimate is needed for fine splits,
  and for splits selected by the reference's values,
  as in [choosing the baseline](../design/probabilistic-framework.md#choosing-the-baseline).

## Justification

The derivation is in [a general rule for volumes](../design/probabilistic-framework.md#a-general-rule-for-volumes).
The identity $\mathrm{NMSE} = \sum_k v_k E_k^2$ holds for any volumes that sum to 1,
as check S confirms, so two axioms are needed to fix them.

- **Axiom A (conditional expectation).** For every orthogonal decomposition,
  $\mathrm{NMSE} = \sum_k v(P_k) E_k^2$ with $\sum_k v(P_k) = 1$,
  and $v(P)$ depends only on $P$, not on how the rest of the space is split.
  This is the law of total expectation,
  $\mathrm{E}[X] = \sum_k \Pr(A_k) \mathrm{E}[X \mid A_k]$,
  with volumes as the probabilities,
  and adjusted errors as the conditional means.
- **Axiom B (independent of the model).** Volumes depend on the subspaces,
  and on the reference, never on the comparison being scored.

From these:

1. Put all of the error in component $k$.
   For any adjusted error of the form $c_k \lVert P_k e \rVert^2 / D$,
   Axiom A then gives $v_k c_k = 1$, so the adjustment factor is $1/v_k$.
2. Volumes are additive, $v(Q + R) = v(Q) + v(R)$ for orthogonal $Q$ and $R$,
   so the adjusted error of a component is a volume-weighted mean,
   over its parts, of their adjusted errors.
   This is the tower property of conditional expectation.
   Vartia (1976) and Diewert (1978) call it consistency in aggregation.
   After one component is split,
   every other component keeps its volume and score, as check T confirms.
3. By Gleason's (1957) theorem, on a Hilbert space of dimension at least 3,
   every such measure on projections has the form $v(P) = \mathrm{tr}(\rho P)$,
   for a positive semidefinite $\rho$ with unit trace.
   Since $\mathrm{tr}(\rho P) = \mathrm{E} \lVert P \varepsilon \rVert^2$,
   for a random field $\varepsilon$ with covariance $\rho$,
   a component's volume is the share of a baseline error that falls in it.

The same $\rho$ gives the volumes of regions, time windows, frequency bands,
and their products, so these are not separate choices,
as shown in [space, time, and frequency](../design/probabilistic-framework.md#space-time-and-frequency).
The RMSE score map already applies this rule to cells.
With $D_{\mathrm{time}}$ and the realized reference,
cell $i$ has volume $w_i \sigma_i^2 / \sum_j w_j \sigma_j^2$,
and its adjusted error is the local error the map scores.
With that baseline,
$1 - E_k^2$ is the mean squared skill score of component $k$ against the benchmark (Murphy, 1988).
Decomposable poverty measures have the same structure,
with population shares as the coefficients (Foster, Greer, & Thorbecke, 1984).

## Considered options

- **Score the raw shares.** Rejected, because shares are extensive,
  so every component scores better than the total, as check P shows.
- **Equal volumes, $1/K$ for $K$ components.** Rejected.
  They satisfy the identity within one decomposition,
  but they fail the first axiom.
  When the variance term is split into two frequency bands,
  the bias term's volume changes from 1/2 to 1/3,
  and its adjusted error from 0.0615 to 0.0923,
  although the bias did not change (check T).
  The coefficients that give the variance term from its bands sum to 4/3,
  as check AA shows.
- **Volumes from the realized reference,
  $\rho \propto \tilde x \tilde x^\top$.** This is one choice of baseline,
  the one-sample estimate of the proposed $\rho$.
  It is adequate for the four-term tree and the cells of a score map.
  It is noisy for small components:
  one day's volume is 0.00003 instead of 1/356 (check U).
  It is biased for components selected by the reference's values:
  the middle two deciles hold 0.3% and 0.6% of the volume (check W).
- **The comparison's own error shares as volumes.** Rejected,
  since they fail the second axiom,
  and every component with any error then has the NMSE as its adjusted error.

## Consequences

- Component scores change.
  The total does not, because $\sum_k v_k E_k^2$ is the NMSE for any volumes.
- The baseline $\rho$ is a modeling choice,
  in the sense of [ADR 0001](0001-probability-theory-as-the-design-rule.md),
  stated alongside the benchmark.
  A pooled estimator of $\rho$ has not been tested.
- A component with a small volume can have a large, noisy adjusted error.
  Its small volume keeps it from dominating the total.
- [ADR 0004](0004-aggregate-then-score-across-analyses-and-datasets.md) applies the same formula across datasets,
  where each dataset's volume is its weight
  and its adjusted error is its own NMSE.
- Checks O–AA support the numbers above.
  They are in `docs/design/experiments/verify_volume.py`, `verify_weights.py`,
  and `verify_factors.py`.

## References

- Diewert, W. E. (1978).
  Superlative index numbers and consistency in aggregation. *Econometrica*,
  46(4), 883–900. https://doi.org/10.2307/1909755
- Foster, J., Greer, J., & Thorbecke, E. (1984).
  A class of decomposable poverty measures. *Econometrica*, 52(3),
  761–766. https://doi.org/10.2307/1913475
- Gleason, A. M. (1957).
  Measures on the closed subspaces of a Hilbert space.
  *Indiana University Mathematics Journal*, 6(4), 885–893.
  https://doi.org/10.1512/iumj.1957.6.56050
- Murphy, A. H. (1988).
  Skill scores based on the mean square error
  and their relationships to the correlation coefficient.
  *Monthly Weather Review*, 116(12), 2417–2424.
  [https://doi.org/10.1175/1520-0493(1988)116<2417:SSBOTM>2.0.CO;2](https://doi.org/10.1175/1520-0493(1988)116<2417:SSBOTM>2.0.CO;2)
- Vartia, Y. O. (1976).
  Ideal log-change index numbers. *Scandinavian Journal of Statistics*, 3(3),
  121–126.
