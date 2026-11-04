"""
Checks for the orthogonal replacements of canonical metrics (docs/design/probabilistic-framework.md).

Reuses the synthetic data of verify_components.py, which must be in the same
directory. Run from an ilamb3 checkout: python verify_suite.py
"""

import warnings

warnings.simplefilter("ignore")

import numpy as np
from verify_components import (
    AREA,
    TAU,
    anomaly,
    b,
    bisector,
    clim,
    d_amp,
    d_cor,
    e,
    e_amp,
    e_tim,
    f,
    ip,
    mean_bias,
    mse,
    pattern,
    seasonal,
    smean,
    tmean,
    x,
)
from verify_components import rng as components_rng

RESULTS = []


def report(name, passed, detail):
    RESULTS.append((name, bool(passed), detail))


# -----------------------------------------------------------------------------------
# J. The anomaly term splits into amplitude and correlation parts, which are orthogonal
#    to each other and to the other five terms: a tree of seven orthogonal terms.
# -----------------------------------------------------------------------------------
a_m = f - tmean(f) - (clim(f) - tmean(f))
a_r = x - tmean(x) - (clim(x) - tmean(x))
same = np.allclose(a_m - a_r, anomaly)
an_amp, an_cor = bisector(a_m, a_r)
leaves = {
    "mean bias": mean_bias,
    "pattern amplitude": d_amp + 0 * e,
    "pattern correlation": d_cor + 0 * e,
    "seasonal amplitude": e_amp,
    "seasonal timing": e_tim,
    "anomaly amplitude": an_amp,
    "anomaly correlation": an_cor,
}
names = list(leaves)
cross = max(
    abs(ip(leaves[p], leaves[q])) / mse for i, p in enumerate(names) for q in names[i + 1 :]
)
total = sum(ip(v_, v_) for v_ in leaves.values())
report(
    "J  anomaly amplitude + correlation; seven orthogonal terms",
    same and cross < 1e-12 and np.isclose(total, mse, rtol=1e-12),
    f"anomaly error = model anomalies - reference anomalies: {same}; largest of the 21 "
    f"cross terms = {cross:.1e} of the MSE; the seven terms sum to the MSE: "
    f"{np.isclose(total, mse, rtol=1e-12)}",
)

# -----------------------------------------------------------------------------------
# K. The seasonal term splits into season means and a within-season remainder (nested
#    subspaces: time-invariant within cell-and-season within cell-and-month).
# -----------------------------------------------------------------------------------
w_season = TAU.groupby("time.season").sum("time")
b_s = (e * TAU).groupby("time.season").sum("time") / w_season
p_s = w_season / TAU.sum()
season_part = (b_s - b).sel(season=e["time.season"]).drop_vars("season")
within = seasonal - season_part
hydro = float(smean((p_s * (b_s - b) ** 2).sum("season")))
report(
    "K  seasonal term = season means + within-season remainder",
    abs(ip(season_part, within)) < 1e-12 * mse
    and np.isclose(ip(season_part, season_part), hydro)
    and np.isclose(ip(season_part, season_part) + ip(within, within), ip(seasonal, seasonal)),
    f"cross term {abs(ip(season_part, within)) / mse:.1e} of the MSE; the season-mean "
    f"part equals sum_s p_s (b_s - b)^2 from the Hydro identity "
    f"({ip(season_part, season_part) / ip(seasonal, seasonal):.1%} of the seasonal term)",
)

# -----------------------------------------------------------------------------------
# L. The MSE of the regional-mean series is the mean bias plus the spatially uniform
#    parts of the seasonal and anomaly terms.
# -----------------------------------------------------------------------------------
series = smean(e)
series_mse = float(tmean(series**2))
uni_s, uni_a = smean(seasonal), smean(anomaly)
parts_l = float(smean(b)) ** 2 + float(tmean(uni_s**2)) + float(tmean(uni_a**2))
local_ok = np.isclose(
    ip(seasonal, seasonal), float(tmean(uni_s**2)) + ip(seasonal - uni_s, seasonal - uni_s)
) and np.isclose(
    ip(anomaly, anomaly), float(tmean(uni_a**2)) + ip(anomaly - uni_a, anomaly - uni_a)
)
report(
    "L  regional-mean series = mean bias + uniform seasonal + uniform anomaly",
    np.isclose(series_mse, parts_l) and local_ok,
    f"series MSE {series_mse:.6f} vs sum {parts_l:.6f}; the seasonal and anomaly terms "
    f"each split orthogonally into uniform and local parts: {local_ok}; the series "
    f"MSE is {series_mse / mse:.1%} of the MSE",
)

# -----------------------------------------------------------------------------------
# M. Area: missed and excess cells are disjoint, so their errors are orthogonal.
# -----------------------------------------------------------------------------------
rng = np.random.default_rng(5)
w = AREA.values.ravel()
i_r = (rng.normal(size=w.size) > 0).astype(float)
i_m = (rng.normal(size=w.size) > 0.3).astype(float)
dd = i_m - i_r
missed, excess = np.minimum(dd, 0), np.maximum(dd, 0)
a_ref = np.sum(w * i_r)
report(
    "M  Area: missed and excess shares are orthogonal",
    np.sum(w * missed * excess) == 0
    and np.isclose(np.sum(w * dd**2) / a_ref, (np.sum(w * missed**2) + np.sum(w * excess**2)) / a_ref),
    f"cross term {np.sum(w * missed * excess):.1f}; missed {np.sum(w * missed**2) / a_ref:.3f} + "
    f"excess {np.sum(w * excess**2) / a_ref:.3f} of the reference area = total "
    f"{np.sum(w * dd**2) / a_ref:.3f}",
)

# -----------------------------------------------------------------------------------
# N. Relationship: the error explained by binned values of a driver is an orthogonal
#    split of the MSE, but it straddles the pattern, seasonal, and anomaly terms.
# -----------------------------------------------------------------------------------
z = x + (components_rng.normal(0, 1, x.shape) * 0.5)  # a driver correlated with the reference
weights = (AREA * TAU).broadcast_like(e)
edges = np.quantile(z.values, np.linspace(0, 1, 11))
label = np.clip(np.searchsorted(edges, z.values, side="right") - 1, 0, 9)
ev, wv = e.values, weights.values
cond = np.zeros_like(ev)
for k in range(10):
    sel = label == k
    cond[sel] = np.sum(ev[sel] * wv[sel]) / np.sum(wv[sel])
cond = e.copy(data=cond)
orth = abs(ip(cond, e - cond)) / mse
captured = {
    k: ip(cond, v_) / ip(v_, v_)
    for k, v_ in {
        "mean bias": mean_bias,
        "pattern": pattern,
        "seasonal": seasonal,
        "anomaly": anomaly,
    }.items()
}
report(
    "N  driver-conditional split is orthogonal but straddles the tree",
    orth < 1e-12 and any(0.05 < c_ < 0.95 for c_ in captured.values()),
    f"explained part {ip(cond, cond) / mse:.1%} of the MSE, cross term with the rest "
    f"{orth:.1e}; it captures "
    + ", ".join(f"{c_:.0%} of the {k} term" for k, c_ in captured.items())
    + " (a sum of whole terms would capture each term entirely or not at all)",
)

if __name__ == "__main__":
    width = max(len(name) for name, _, _ in RESULTS)
    for name, passed, detail in RESULTS:
        print(f"{'PASS' if passed else 'FAIL'}  {name:<{width}}  {detail}")
