"""
Checks for the orthogonal decompositions of the MSE behind the Spatial Distribution
and Seasonal Cycle scores (docs/design/probabilistic-framework.md).

Synthetic data on a 10-degree grid with 3.5 years of monthly values (the last year
is incomplete), weighted with ilamb3.dataset.compute_cell_measures and
compute_time_measures. Run from an ilamb3 checkout: python verify_components.py
"""

import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import xarray as xr

import ilamb3.dataset as dset
from ilamb3.analysis import scoring

RESULTS = []
NLAT, NLON = 18, 36
SPACE = ("lat", "lon")


def report(name, passed, detail):
    RESULTS.append((name, bool(passed), detail))


def _centers(lo, hi, n):
    edges = np.linspace(lo, hi, n + 1)
    return 0.5 * (edges[1:] + edges[:-1])


TIME = pd.to_datetime([f"{2000 + k // 12}-{k % 12 + 1:02d}-15" for k in range(42)])
LAT, LON = _centers(-90, 90, NLAT), _centers(-180, 180, NLON)
TEMPLATE = xr.DataArray(
    np.zeros((len(TIME), NLAT, NLON)),
    coords={"time": TIME, "lat": LAT, "lon": LON},
    dims=("time", "lat", "lon"),
    name="v",
    attrs={"units": "1"},
)
AREA = dset.compute_cell_measures(TEMPLATE)
TAU = dset.compute_time_measures(TEMPLATE)
MONTH = TEMPLATE["time"].dt.month


def tmean(x):
    return (x * TAU).sum("time") / TAU.sum()


def smean(x):
    return (x * AREA).sum(SPACE) / AREA.sum()


def ip(a, b):
    """The inner product whose norm is the MSE: an area- and time-weighted mean."""
    return float(smean(tmean(a * b)))


def clim(x):
    """Time-weighted monthly climatology, broadcast back to the time axis."""
    c = (x * TAU).groupby("time.month").sum("time") / TAU.groupby("time.month").sum("time")
    return c.sel(month=x["time.month"]).drop_vars("month")


def spatial(values):
    return xr.DataArray(values, coords={"lat": LAT, "lon": LON}, dims=SPACE)


rng = np.random.default_rng(3)
coslat = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, NLON))
phase_r = 2 * np.pi * MONTH / 12
x_mean = spatial(5 + 3 * coslat + rng.normal(0, 0.5, (NLAT, NLON)))
x_amp = spatial(1 + 2 * coslat + rng.uniform(0, 0.5, (NLAT, NLON)))
x = x_mean + x_amp * np.sin(phase_r) + TEMPLATE.copy(data=rng.normal(0, 0.7, TEMPLATE.shape))
lag = spatial(rng.uniform(0, 2, (NLAT, NLON)))  # months
f = (
    0.3
    + 0.55 * x_mean
    + spatial(rng.normal(0, 0.3, (NLAT, NLON)))
    + 1.3 * x_amp * np.sin(phase_r - 2 * np.pi * lag / 12)
    + 0.3 * x_amp * np.sin(2 * phase_r)
    + TEMPLATE.copy(data=rng.normal(0, 0.9, TEMPLATE.shape))
)

# -----------------------------------------------------------------------------------
# A. Four nested orthogonal projections: global constant, time-invariant fields,
#    fields that depend only on cell and calendar month, everything.
# -----------------------------------------------------------------------------------
e = f - x
b = tmean(e)
mean_bias = smean(b) + 0 * e
pattern = (b - smean(b)) + 0 * e
c = e - b
seasonal = clim(c)
anomaly = c - seasonal
parts = {"mean bias": mean_bias, "pattern": pattern, "seasonal": seasonal, "anomaly": anomaly}
mse = ip(e, e)
names = list(parts)
cross = max(
    abs(ip(parts[a], parts[b_])) / mse
    for i, a in enumerate(names)
    for b_ in names[i + 1 :]
)
total = sum(ip(p, p) for p in parts.values())
seas_u = c.groupby("time.month").mean("time").sel(month=c["time.month"]).drop_vars("month")
cross_u = abs(ip(seas_u, c - seas_u)) / mse
report(
    "A  four-term split is orthogonal, with an incomplete final year",
    cross < 1e-12 and np.isclose(total, mse, rtol=1e-12),
    f"largest of the 6 cross terms = {cross:.1e} of the MSE; sum of squares = MSE: "
    f"{np.isclose(total, mse, rtol=1e-12)}; with an unweighted climatology, as Seasonal "
    f"Cycle computes it, the seasonal-anomaly cross term is {cross_u:.1e} of the MSE",
)

# -----------------------------------------------------------------------------------
# B. The pattern term is the Taylor law of cosines; three ways to split it.
# -----------------------------------------------------------------------------------
mbar, xbar = tmean(f), tmean(x)
mp, rp = mbar - smean(mbar), xbar - smean(xbar)
sm, sr = float(np.sqrt(smean(mp**2))), float(np.sqrt(smean(rp**2)))
R = float(smean(mp * rp)) / (sm * sr)
d = mp - rp
var_b = float(smean(d**2))
u, v = rp / sr, mp / sm
law = np.isclose(var_b, sm**2 + sr**2 - 2 * sm * sr * R) and np.allclose(d, b - smean(b))

n = (u + v) / float(np.sqrt(smean((u + v) ** 2)))
d_amp = float(smean(d * n)) * n
d_cor = d - d_amp
bis = (
    abs(float(smean(d_amp * d_cor))) < 1e-12 * var_b
    and np.isclose(float(smean(d_amp**2)), (sm - sr) ** 2 * (1 + R) / 2)
    and np.isclose(float(smean(d_cor**2)), (sm + sr) ** 2 * (1 - R) / 2)
)
d_cb = (sm - R * sr) * v
d_ps = d - d_cb
mur = (
    abs(float(smean(d_cb * d_ps))) < 1e-12 * var_b
    and np.isclose(float(smean(d_cb**2)), (sm - R * sr) ** 2)
    and np.isclose(float(smean(d_ps**2)), sr**2 * (1 - R**2))
)
g1, g2 = (sm - sr) * v, sr * (v - u)
gupta_cross = 2 * float(smean(g1 * g2))
report(
    "B  pattern term: law of cosines; bisector and Murphy splits are orthogonal",
    law and bis and mur and not np.isclose(gupta_cross, 0),
    f"Var(b) = s_m^2 + s_r^2 - 2 s_m s_r R: {law}; bisector split orthogonal with the "
    f"stated terms: {bis}; Murphy split orthogonal: {mur}; Gupta's two vectors have a "
    f"cross term of {gupta_cross / var_b:.1%} of Var(b), so that split is additive but "
    f"not orthogonal",
)


# -----------------------------------------------------------------------------------
# C. Scale, shape, and sequence: R <= R_q <= 1 with area weights, and R_q = 1 when the
#    two fields have the same distribution up to location and scale.
# -----------------------------------------------------------------------------------
def quantile_pairs(a, c_, w):
    """Weighted quantile functions of a and c_ on a common partition of [0, 1]."""
    w = w / w.sum()
    ia, ic = np.argsort(a), np.argsort(c_)
    pa, pc = np.cumsum(w[ia]), np.cumsum(w[ic])
    right = np.union1d(pa, pc)
    left = np.concatenate([[0.0], right[:-1]])
    mid = 0.5 * (left + right)
    qa = a[ia][np.minimum(np.searchsorted(pa, mid), a.size - 1)]
    qc = c_[ic][np.minimum(np.searchsorted(pc, mid), c_.size - 1)]
    return qa, qc, right - left


def scale_shape_sequence(a, c_, w):
    w = w / w.sum()
    ma, mc = np.sum(w * a), np.sum(w * c_)
    sa, sc = np.sqrt(np.sum(w * (a - ma) ** 2)), np.sqrt(np.sum(w * (c_ - mc) ** 2))
    r = np.sum(w * (a - ma) * (c_ - mc)) / (sa * sc)
    qa, qc, dp = quantile_pairs(a, c_, w)
    rq = np.sum(dp * (qa - ma) * (qc - mc)) / (sa * sc)
    var_omega = np.sum(dp * ((qa - ma) - (qc - mc)) ** 2)
    return sa, sc, r, rq, var_omega


w = AREA.values.ravel()
sa, sc, r, rq, var_omega = scale_shape_sequence(mbar.values.ravel(), xbar.values.ravel(), w)
order_ok = r <= rq + 1e-12 and rq <= 1 + 1e-12
dist_ok = np.isclose(var_omega, (sa - sc) ** 2 + 2 * sa * sc * (1 - rq))
x_eq = rng.normal(0, 1, 500) ** 3
m_eq = 4 + 0.7 * rng.permutation(x_eq)
sa2, sc2, r2, rq2, vo2 = scale_shape_sequence(m_eq, x_eq, np.ones(500))
same_shape = np.isclose(rq2, 1) and np.isclose(vo2, (sa2 - sc2) ** 2)
report(
    "C  scale + shape + sequence = pattern term, all nonnegative",
    order_ok and dist_ok and same_shape,
    f"area-weighted: R = {r:.4f} <= R_q = {rq:.4f} <= 1; distribution term = "
    f"(s_m - s_r)^2 + 2 s_m s_r (1 - R_q): {dist_ok}; a shuffled affine copy of a skewed "
    f"field has R_q = {rq2:.6f}, so the split reduces to Gupta's",
)

# -----------------------------------------------------------------------------------
# D. The Taylor score is not a function of the pattern MSE.
# -----------------------------------------------------------------------------------
N = 2000
r_c = rng.normal(0, 1, N)
r_c -= r_c.mean()
z = rng.normal(0, 1, N)
z -= z.mean()
z -= (z @ r_c) / (r_c @ r_c) * r_c
z *= r_c.std() / z.std()


def taylor_and_nmse(m):
    """Spatial Distribution's formula, and the pattern MSE over the reference variance."""
    s_hat = m.std() / r_c.std()
    corr = np.corrcoef(r_c, m)[0, 1]
    score = 4 * (1 + corr) / ((s_hat + 1 / s_hat) ** 2 * 2)
    return score, np.mean((m - m.mean() - r_c) ** 2) / r_c.var()


s1, e1 = taylor_and_nmse(0.5 * r_c)
s2, e2 = taylor_and_nmse(0.8 * r_c + 0.6 * z)
grid = np.linspace(0.05, 2, 391)
best_mse = grid[np.argmin(grid**2 + 1 - 2 * grid * 0.6)]
best_taylor = grid[np.argmax(2 * (1 + 0.6) / (grid + 1 / grid) ** 2)]
report(
    "D  Taylor score ranks against the pattern MSE",
    e1 < e2 and s1 < s2 and np.isclose(best_mse, 0.6) and np.isclose(best_taylor, 1),
    f"model 1 (s_hat 0.5, R 1): pattern NMSE {e1:.3f}, Taylor score {s1:.3f}; model 2 "
    f"(s_hat 1, R 0.8): pattern NMSE {e2:.3f}, Taylor score {s2:.3f}; for R = 0.6 the "
    f"pattern MSE is smallest at s_hat = {best_mse:.2f}, the Taylor score largest at "
    f"s_hat = {best_taylor:.2f}",
)

# -----------------------------------------------------------------------------------
# E. The seasonal term is a law of cosines in each cell; amplitude and timing split.
# -----------------------------------------------------------------------------------
def bisector(model, ref):
    """Per-cell bisector split of model - ref into amplitude and correlation parts."""
    sm, sr = np.sqrt(tmean(model**2)), np.sqrt(tmean(ref**2))
    u, v = ref / sr, model / sm
    n = (u + v) / np.sqrt(tmean((u + v) ** 2))
    amp = tmean((model - ref) * n) * n
    return amp, (model - ref) - amp


s_m, s_r = clim(f) - tmean(f), clim(x) - tmean(x)
same = np.allclose(s_m - s_r, seasonal)
vm, vr, cv = tmean(s_m**2), tmean(s_r**2), tmean(s_m * s_r)
a_m, a_r = np.sqrt(vm), np.sqrt(vr)
Rs = cv / (a_m * a_r)
cell_law = np.allclose(tmean(seasonal**2), vm + vr - 2 * cv)
amp = float(smean((a_m - a_r) ** 2 * (1 + Rs) / 2))
timing = float(smean((a_m + a_r) ** 2 * (1 - Rs) / 2))
seas = ip(seasonal, seasonal)
e_amp, e_tim = bisector(s_m, s_r)
orth = abs(ip(e_amp, e_tim)) < 1e-12 * seas
report(
    "E  seasonal term: law of cosines per cell; amplitude + timing, orthogonal",
    same and cell_law and np.isclose(amp + timing, seas) and orth,
    f"seasonal error = model cycle - reference cycle: {same}; per-cell law of cosines: "
    f"{cell_law}; amplitude {amp / seas:.1%} + timing {timing / seas:.1%} of the seasonal "
    f"term; the two error fields are orthogonal: {orth}",
)

# -----------------------------------------------------------------------------------
# F. For a sinusoidal cycle, R_s = cos(2 pi lag / 12); harmonics split it further.
# -----------------------------------------------------------------------------------
months = np.arange(12)
errs_eq, errs_tau = [], []
days = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
for lag_m in range(7):
    ref_c = np.sin(2 * np.pi * months / 12)
    mod_c = 1.7 * np.sin(2 * np.pi * (months - lag_m) / 12)
    errs_eq.append(abs(np.corrcoef(ref_c, mod_c)[0, 1] - np.cos(2 * np.pi * lag_m / 12)))
    p = days / days.sum()
    rc, mc = ref_c - p @ ref_c, mod_c - p @ mod_c
    r_tau = (p @ (rc * mc)) / np.sqrt((p @ rc**2) * (p @ mc**2))
    errs_tau.append(abs(r_tau - np.cos(2 * np.pi * lag_m / 12)))
ref_c = np.sin(2 * np.pi * months / 12) + 0.4 * np.cos(4 * np.pi * months / 12)
mod_c = 1.3 * np.sin(2 * np.pi * (months - 1.5) / 12) + 0.1 * np.sin(4 * np.pi * months / 12)
X, M = np.fft.rfft(ref_c), np.fft.rfft(mod_c)
k_w = np.array([0, 2, 2, 2, 2, 2, 1]) / 12**2  # Parseval weights for 12 points
phi = np.angle(M * np.conj(X))
amp_k = k_w * (np.abs(M) - np.abs(X)) ** 2 * np.cos(phi / 2) ** 2
tim_k = k_w * (np.abs(M) + np.abs(X)) ** 2 * np.sin(phi / 2) ** 2
target = np.mean((mod_c - mod_c.mean() - (ref_c - ref_c.mean())) ** 2)
harm_ok = np.isclose(amp_k[1:].sum() + tim_k[1:].sum(), target)
report(
    "F  sinusoid: R_s = cos(2 pi lag / 12); per-harmonic amplitude and phase terms",
    max(errs_eq) < 1e-12 and harm_ok,
    f"equal month weights: largest error {max(errs_eq):.1e}; with month lengths as "
    f"weights: largest error {max(errs_tau):.1e}; amplitude and phase terms of each "
    f"harmonic sum to the seasonal MSE: {harm_ok}",
)


# -----------------------------------------------------------------------------------
# G. The phase shift is not a function of the seasonal MSE.
# -----------------------------------------------------------------------------------
def shift_score_and_nmse(amplitude, lag_m):
    ref_c = np.sin(2 * np.pi * months / 12)
    mod_c = amplitude * np.sin(2 * np.pi * (months - lag_m) / 12)
    shift = int(np.argmax(mod_c)) - int(np.argmax(ref_c))
    shift = shift - 12 if shift > 6 else shift + 12 if shift < -6 else shift
    # The Seasonal Cycle score, linear in the absolute shift
    stats = scoring.statistics(xr.DataArray(float(shift)), 6.0, power=1)
    score = float(scoring.score(stats, "Linear"))
    return score, np.mean((mod_c - ref_c) ** 2) / np.mean(ref_c**2)


a_score, a_err = shift_score_and_nmse(2.5, 0)
b_score, b_err = shift_score_and_nmse(1.0, 2)
c_score, c_err = shift_score_and_nmse(2.0, 0)
report(
    "G  phase-shift score ranks against the seasonal MSE",
    a_err > b_err and a_score > b_score and np.isclose(c_err, b_err) and c_score != b_score,
    f"amplitude x2.5, no lag: seasonal NMSE {a_err:.2f}, shift score {a_score:.3f}; "
    f"amplitude x1, 2-month lag: NMSE {b_err:.2f}, score {b_score:.3f}; amplitude x2, no "
    f"lag: NMSE {c_err:.2f} (the same as the lagged model), score {c_score:.3f}",
)

# -----------------------------------------------------------------------------------
# H. The full tree sums back to the MSE.
# -----------------------------------------------------------------------------------
leaves = {
    "mean bias": ip(mean_bias, mean_bias),
    "pattern amplitude": float(smean(d_amp**2)),
    "pattern correlation": float(smean(d_cor**2)),
    "seasonal amplitude": amp,
    "seasonal timing": timing,
    "anomaly": ip(anomaly, anomaly),
}
report(
    "H  six leaves sum to the MSE",
    np.isclose(sum(leaves.values()), mse, rtol=1e-12),
    "; ".join(f"{k} {val / mse:.1%}" for k, val in leaves.items()),
)

# -----------------------------------------------------------------------------------
# I. Weights on the bisector parts: equal weights (the MSE) are optimal at s_hat = R;
#    only dropping the correlation part moves the optimum to s_hat = 1.
# -----------------------------------------------------------------------------------
R_fix = 0.6
grid = np.linspace(0.01, 2, 19901)


def optimum(alpha, beta):
    loss = alpha * (grid - 1) ** 2 * (1 + R_fix) / 2 + beta * (grid + 1) ** 2 * (1 - R_fix) / 2
    return grid[np.argmin(loss)]


def formula(alpha, beta):
    num = alpha * (1 + R_fix) - beta * (1 - R_fix)
    return num / (alpha * (1 + R_fix) + beta * (1 - R_fix))


cases = [(1, 1), (2, 1), (5, 1), (1, 0)]
report(
    "I  weighting the amplitude part moves the optimum toward s_hat = 1",
    all(abs(optimum(a_, b_) - formula(a_, b_)) < 1e-3 for a_, b_ in cases)
    and abs(optimum(1, 1) - R_fix) < 1e-3
    and abs(optimum(1, 0) - 1) < 1e-3,
    f"R = {R_fix}: optimal s_hat for weights (amplitude, correlation) "
    + ", ".join(f"({a_}, {b_}) -> {optimum(a_, b_):.3f}" for a_, b_ in cases),
)

if __name__ == "__main__":
    width = max(len(name) for name, _, _ in RESULTS)
    for name, passed, detail in RESULTS:
        print(f"{'PASS' if passed else 'FAIL'}  {name:<{width}}  {detail}")
