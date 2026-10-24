"""
Quick verification experiments for the fixes in docs/design/probabilistic-framework.md.

Run from an ilamb3 checkout: python verify_fixes.py

Every experiment uses synthetic data. The gridded ones use the grid of
verify_components.py, which must be in the same directory: cells weighted by
ilamb3.dataset.compute_cell_measures and times by ilamb3.dataset.compute_time_measures,
so the identities are checked with the weights ilamb3 uses.
"""

import warnings

warnings.simplefilter("ignore")

import numpy as np
import xarray as xr
from verify_components import (
    AREA,
    LAT,
    MONTH,
    NLAT,
    NLON,
    SPACE,
    TAU,
    TEMPLATE,
    spatial,
    tmean,
)

from ilamb3.analysis import ALL_ANALYSES
from ilamb3.analysis.relationship import Relationship

RESULTS = []


def report(name, passed, detail):
    RESULTS.append((name, bool(passed), detail))


def smean(x, w=AREA):
    return (x * w).sum(SPACE) / w.sum(SPACE)


def seasonal_cycle(mean, amp, phase_months):
    return mean + amp * np.sin(2 * np.pi * (MONTH - phase_months) / 12)


def reference(seed=0):
    rng = np.random.default_rng(seed)
    coslat = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, NLON))
    mean = spatial(5 + 3 * coslat + rng.normal(0, 0.5, (NLAT, NLON)))
    amp = spatial(1 + 2 * coslat + rng.uniform(0, 0.5, (NLAT, NLON)))
    noise = TEMPLATE.copy(data=rng.normal(0, 0.7, TEMPLATE.shape))
    return seasonal_cycle(mean, amp, 0), mean, amp, noise


X, X_MEAN, X_AMP, X_NOISE = reference()
X = X + X_NOISE


def model(seed, bias=0.0, pattern=0.0, amp_scale=1.0, phase=0.0, noise=0.7, rho=0.5):
    """A model whose error in each component is set separately."""
    rng = np.random.default_rng(seed)
    pattern_field = spatial(rng.normal(0, 1, (NLAT, NLON)))
    own = TEMPLATE.copy(data=rng.normal(0, 1, TEMPLATE.shape))
    anomalies = rho * X_NOISE + np.sqrt(1 - rho**2) * noise * own
    return (
        seasonal_cycle(X_MEAN + bias + pattern * pattern_field, amp_scale * X_AMP, phase)
        + anomalies
    )


def decompose(f, x, w=AREA):
    """Split the pooled MSE into mean bias, bias pattern, seasonal cycle, and anomalies."""
    e = f - x
    b = tmean(e)
    centered = e - b
    clim = (centered * TAU).groupby("time.month").sum("time") / TAU.groupby(
        "time.month"
    ).sum("time")
    c = clim.sel(month=centered["time.month"]).drop_vars("month")
    a = centered - c
    D = smean(tmean((x - tmean(x)) ** 2), w)
    parts = {
        "mean": smean(b, w) ** 2 / D,
        "pattern": smean((b - smean(b, w)) ** 2, w) / D,
        "seasonal": smean(tmean(c**2), w) / D,
        "anomaly": smean(tmean(a**2), w) / D,
        "total": smean(tmean(e**2), w) / D,
    }
    return {k: float(v) for k, v in parts.items()}, dict(e=e, b=b, c=c, a=a, D=D)


def close(a, b, rtol=1e-9):
    return np.allclose(a, b, rtol=rtol, atol=1e-12)


# ---------------------------------------------------------------------------------
# Fix 1: one normalizer. The total variance is the pooled within-cell variance in time
# plus the spatial variance of the time means (law of total variance).
# ---------------------------------------------------------------------------------
xbar = smean(tmean(X))
d_tot = float(smean(tmean((X - xbar) ** 2)))
d_time = float(smean(tmean((X - tmean(X)) ** 2)))
d_space = float(smean((tmean(X) - xbar) ** 2))
report(
    "1  normalizer: D_total = D_time + D_space",
    close(d_tot, d_time + d_space),
    f"D_total={d_tot:.6g}, D_time+D_space={d_time + d_space:.6g}",
)


# ---------------------------------------------------------------------------------
# Fix 2: the pattern term is the Taylor quantity only when the spatial statistics are
# area-weighted (the identities themselves are checks A and B of verify_components.py).
# ---------------------------------------------------------------------------------
F = model(1, bias=0.4, pattern=0.8, amp_scale=1.2, phase=1.0)
parts, aux = decompose(F, X)
b = aux["b"]
mf, mx = tmean(F), tmean(X)
var_b = float(smean((b - smean(b)) ** 2))
# ilamb3's Spatial Distribution statistics, which are not area-weighted
sd = ALL_ANALYSES["spatial_distribution"]("v")
table = sd(mx.to_dataset(name="v"), mf.to_dataset(name="v"))[0].set_index("name")
s_hat = float(table.loc["Normalized Standard Deviation", "value"])
r_u = float(table.loc["Correlation", "value"])
unweighted = float(mx.std()) ** 2 * (s_hat**2 + 1 - 2 * s_hat * r_u)
gap = abs(unweighted - var_b) / var_b
report(
    "2  Spatial Distribution misses the pattern term without area weights",
    gap > 1e-3,
    f"s_r^2 (s_hat^2 + 1 - 2 s_hat R) from Spatial Distribution's statistics is off "
    f"from the area-weighted Var(b) by {gap:.1%}",
)


# ---------------------------------------------------------------------------------
# Fix 3: the centered error splits into its monthly climatology and the anomalies, and a
# pure phase shift lands entirely in the seasonal term.
# ---------------------------------------------------------------------------------
cell_ok = bool(
    np.allclose(
        tmean((aux["e"] - aux["b"]) ** 2), tmean(aux["c"] ** 2) + tmean(aux["a"] ** 2)
    )
)
shift_parts = []
for k in range(7):
    Fk = X - seasonal_cycle(0, X_AMP, 0) + seasonal_cycle(0, X_AMP, k)
    shift_parts.append(decompose(Fk, X)[0])
anomaly_zero = all(p["anomaly"] < 1e-20 for p in shift_parts)
# With an incomplete final year, a shifted cycle also has a period-mean bias
bias_share = max((p["mean"] + p["pattern"]) / p["total"] for p in shift_parts[1:])
seasonal_share = min(p["seasonal"] / p["total"] for p in shift_parts[1:])
monotone = all(
    shift_parts[i]["seasonal"] < shift_parts[i + 1]["seasonal"] for i in range(6)
)
report(
    "3  seasonal split; a pure phase shift is seasonal error",
    cell_ok and anomaly_zero and seasonal_share > 0.95 and monotone,
    f"per-cell identity: {cell_ok}; anomaly term for shifts 0-6 months is 0: {anomaly_zero}; "
    f"seasonal share >= {seasonal_share:.2%}, the rest a period-mean bias from the "
    f"incomplete final year (<= {bias_share:.2%}); seasonal term increases with the "
    f"shift: {monotone}",
)

# ---------------------------------------------------------------------------------
# Fix 4: the four terms sum to the normalized MSE; scoring the total ranks models the same
# way for every decreasing score function, while averaging component scores does not.
# ---------------------------------------------------------------------------------
models = []
for i in range(20):
    r = np.random.default_rng(100 + i)
    models.append(
        decompose(
            model(
                100 + i,
                bias=r.uniform(-1, 1),
                pattern=r.uniform(0, 1.5),
                amp_scale=r.uniform(0.5, 1.5),
                phase=r.uniform(0, 3),
                noise=r.uniform(0.2, 1.5),
                rho=r.uniform(0, 0.9),
            ),
            X,
        )[0]
    )
sums_ok = all(
    close(p["mean"] + p["pattern"] + p["seasonal"] + p["anomaly"], p["total"])
    for p in models
)
E = np.sqrt([p["total"] for p in models])
truth = np.argsort(E)
score_fns = {
    "exp(-0.25E)": lambda e: np.exp(-0.25 * e),
    "exp(-E)": lambda e: np.exp(-e),
    "exp(-4E)": lambda e: np.exp(-4 * e),
    "1/(1+E)": lambda e: 1 / (1 + e),
}
same_order = all(np.array_equal(np.argsort(-s(E)), truth) for s in score_fns.values())
comp = np.sqrt([[p[k] for k in ("mean", "pattern", "seasonal", "anomaly")] for p in models])


def composite(alpha):
    return np.exp(-alpha * comp).mean(axis=1)


def discordant(u, v):
    n = len(u)
    return sum(
        np.sign(u[i] - u[j]) != np.sign(v[i] - v[j]) for i in range(n) for j in range(i + 1, n)
    )


flips = discordant(composite(0.25), composite(4))
report(
    "4  total = sum of terms; ranking invariant only when the total is scored",
    sums_ok and same_order and flips > 0,
    f"sum of terms equals the total for 20 models: {sums_ok}; same ranking under "
    f"{', '.join(score_fns)}: {same_order}; averaging the four component scores reorders "
    f"{flips} of 190 pairs between alpha = 0.25 and alpha = 4",
)


# ---------------------------------------------------------------------------------
# Fix 5 (uncertainty): subtracting the reference error variance once is unbiased and
# keeps the terms additive; clipping each term does not add up.
# ---------------------------------------------------------------------------------
rng = np.random.default_rng(5)
cells, nt, trials = 50, 48, 400
sigma_u = rng.uniform(0.2, 0.8, cells)
months = np.arange(nt) % 12
truth = 3 + 2 * np.sin(2 * np.pi * months / 12)[None, :] + rng.normal(0, 1, (cells, nt))
corrected = []
for _ in range(trials):
    xr_ = truth + rng.normal(0, 1, (cells, nt)) * sigma_u[:, None]
    D10 = xr_.var(axis=1).mean()
    mse = ((truth - xr_) ** 2).mean()
    corrected.append((mse - (sigma_u**2).mean()) / D10)
corrected = np.array(corrected)
unbiased = abs(corrected.mean()) < 3 * corrected.std() / np.sqrt(trials)
f10 = truth + 0.6 + rng.normal(0, 0.8, (cells, nt))
x10 = truth + rng.normal(0, 1, (cells, nt)) * sigma_u[:, None]
e10 = f10 - x10
u = sigma_u[:, None]
b10 = e10.mean(axis=1, keepdims=True)
clip_terms = np.clip(np.abs(b10) - u, 0, None) ** 2 + (
    np.clip(np.abs(e10 - b10) - u, 0, None) ** 2
).mean(axis=1, keepdims=True)
clip_total = (np.clip(np.abs(e10) - u, 0, None) ** 2).mean(axis=1, keepdims=True)
gap = np.median(np.abs(clip_terms - clip_total) / clip_total)
lin_terms = (b10**2 - 0) + ((e10 - b10) ** 2).mean(axis=1, keepdims=True) - u**2
lin_total = (e10**2).mean(axis=1, keepdims=True) - u**2
report(
    "5  uncertainty: subtract the error variance once, not clip each term",
    unbiased and np.allclose(lin_terms, lin_total) and gap > 0.01,
    f"perfect model: corrected NMSE averages {corrected.mean():+.4f} "
    f"(SE {corrected.std() / np.sqrt(trials):.4f}); subtracting keeps the terms additive: "
    f"{np.allclose(lin_terms, lin_total)}; clipping each term misses the clipped total by a "
    f"median {gap:.0%}",
)


# ---------------------------------------------------------------------------------
# Fix 6: mass weighting. The identities hold for any positive weights.
# ---------------------------------------------------------------------------------
w_mass = AREA * np.abs(tmean(X))
pm, am = decompose(F, X, w_mass)
bm = am["b"]
mass_ok = close(pm["mean"] + pm["pattern"] + pm["seasonal"] + pm["anomaly"], pm["total"])
mass_split = close(
    float(smean(bm**2, w_mass)),
    float(smean(bm, w_mass)) ** 2 + float(smean((bm - smean(bm, w_mass)) ** 2, w_mass)),
)
report(
    "6  mass weighting: same weights in every term",
    mass_ok and mass_split,
    f"terms sum to the total with area x reference-mean weights: {mass_ok}; bias split: {mass_split}",
)


# ---------------------------------------------------------------------------------
# Fix 7 (Hydro): seasonal biases contain the annual bias; with one normalizer, the
# seasonal spread is a coarser version of the seasonal term.
# ---------------------------------------------------------------------------------
e = aux["e"]
p_s = TAU.groupby("time.season").sum("time") / TAU.sum()
b_s = (e * TAU).groupby("time.season").sum("time") / TAU.groupby("time.season").sum("time")
mean_ok = bool(np.allclose((p_s * b_s).sum("season"), b))
n_hydro = float(smean((p_s * (b_s - b) ** 2).sum("season")) / aux["D"])
report(
    "7  Hydro: annual bias inside the seasonal biases",
    mean_ok and n_hydro <= parts["seasonal"] + 1e-12,
    f"sum p_s b_s = b: {mean_ok}; seasonal spread {n_hydro:.4g} <= seasonal term "
    f"{parts['seasonal']:.4g} (the split of sum p_s b_s^2 is check K of verify_suite.py)",
)


# ---------------------------------------------------------------------------------
# Fix 8 (Dispersion): Hodson's distribution and sequence terms split the centered MSE.
# ---------------------------------------------------------------------------------
rng = np.random.default_rng(8)
n, ok_id, ok_pos = 48, True, True
for _ in range(2000):
    x = rng.normal(0, 1, n)
    f = 0.5 * x + rng.normal(rng.uniform(-1, 1), rng.uniform(0.1, 2), n)
    e5 = f - x
    omega = np.sort(f) - np.sort(x)
    dist, var_e = omega.var(), e5.var()
    seq = var_e - dist
    ok_id &= np.isclose(e5.mean(), omega.mean())
    ok_pos &= seq >= -1e-12
x = rng.normal(0, 1, n)
perm = rng.permutation(x)
perm_ok = np.isclose((perm - x).mean(), 0) and np.isclose(
    (np.sort(perm) - np.sort(x)).var(), 0
)
scaled = x.mean() + 1.7 * (x - x.mean())
scaled_ok = np.isclose((scaled - x).var() - (np.sort(scaled) - np.sort(x)).var(), 0)
report(
    "8  Dispersion: distribution + sequence = centered MSE",
    ok_id and ok_pos and perm_ok and scaled_ok,
    f"means agree and sequence >= 0 in 2000 random cases: {ok_id and ok_pos}; "
    f"a time permutation of the reference has no bias or distribution error: {perm_ok}; "
    f"scaled anomalies have no sequence error: {scaled_ok}",
)


# ---------------------------------------------------------------------------------
# Fix 9 (Time Series): the same decomposition on one regional series, where the centered
# part is the Taylor quantity in time.
# ---------------------------------------------------------------------------------
fs, xs = smean(F), smean(X)
ets = fs - xs
var_e = float(tmean((ets - tmean(ets)) ** 2))
sf, sx = (float(np.sqrt(tmean((s - tmean(s)) ** 2))) for s in (fs, xs))
cv = float(tmean((fs - tmean(fs)) * (xs - tmean(xs))))
report(
    "9  Time Series: centered MSE of a series = Taylor quantity in time",
    close(var_e, sf**2 + sx**2 - 2 * cv),
    f"Var(e) = s_m^2 + s_r^2 - 2 s_m s_r R for the regional mean series: "
    f"{close(var_e, sf**2 + sx**2 - 2 * cv)}",
)


# ---------------------------------------------------------------------------------
# Fix 10 (Area): the mean of the two overlap ratios can rank more misclassified area
# higher (the misclassified area as a squared indicator error is check M of
# verify_suite.py).
# ---------------------------------------------------------------------------------
cases = {"m1": (50, 10, 10), "m2": (60, 0, 22)}  # overlap, missed, excess; reference area 60
current = {k: 0.5 * (o / (o + m) + o / (o + x)) for k, (o, m, x) in cases.items()}
fixed = {k: (m + x) / (o + m) for k, (o, m, x) in cases.items()}
reversal = current["m2"] > current["m1"] and fixed["m2"] > fixed["m1"]
report(
    "10 Area: the mean of two overlap ratios ranks more misclassified area higher",
    reversal,
    f"with a reference area of 60, m2 misclassifies 22 vs m1's 20, yet the current mean of ratios scores m2 "
    f"higher ({current['m2']:.3f} vs {current['m1']:.3f}); the fixed error ranks m1 first "
    f"({fixed['m1']:.3f} vs {fixed['m2']:.3f})",
)


# ---------------------------------------------------------------------------------
# Fix 11 (Runoff Sensitivity): pooling raw errors in %/% and %/K makes the ranking depend
# on units; normalizing each sensitivity first does not.
# ---------------------------------------------------------------------------------
rng = np.random.default_rng(11)
shape = (30, 3, 4)  # basin, forcing, member
x_p = rng.normal(1.5, 0.4, shape)  # %/%
x_t = rng.normal(-3.0, 1.0, shape)  # %/K
errors = {
    "A": (rng.normal(0, 0.6, shape), rng.normal(0, 0.3, shape)),
    "B": (rng.normal(0, 0.2, shape), rng.normal(0, 1.2, shape)),
}


def pooled_raw(ep, et):
    return np.mean(np.concatenate([ep.ravel() ** 2, et.ravel() ** 2]))


def normalized(ep, et, xp, xt):
    return 0.5 * (np.mean(ep**2) / xp.var() + np.mean(et**2) / xt.var())


raw_k = {m: pooled_raw(ep, et) for m, (ep, et) in errors.items()}
raw_frac = {m: pooled_raw(ep, et / 100) for m, (ep, et) in errors.items()}  # %/K -> 1/K
fix_k = {m: normalized(ep, et, x_p, x_t) for m, (ep, et) in errors.items()}
fix_frac = {m: normalized(ep, et / 100, x_p, x_t / 100) for m, (ep, et) in errors.items()}
raw_flips = (raw_k["A"] < raw_k["B"]) != (raw_frac["A"] < raw_frac["B"])
fix_same = all(np.isclose(fix_k[m], fix_frac[m]) for m in errors)
report(
    "11 Runoff Sensitivity: normalize each sensitivity before pooling",
    raw_flips and fix_same,
    f"raw pooling ranks {'A' if raw_k['A'] < raw_k['B'] else 'B'} first in %/K and "
    f"{'A' if raw_frac['A'] < raw_frac['B'] else 'B'} first in 1/K; normalized values "
    f"unchanged by the unit change: {fix_same}",
)


# ---------------------------------------------------------------------------------
# Fix 12 (Relationship): the current error responds to a pure bias in the dependent
# variable; centering each response curve removes it and keeps errors in the shape.
# ---------------------------------------------------------------------------------
def rel_errors(dep_shift=0.0, slope=1.0):
    rng = np.random.default_rng(7)
    ind = xr.DataArray(rng.uniform(0, 10, 5000), dims="n")
    dep = 2 + 0.5 * ind + rng.normal(0, 0.3, 5000)
    ref = Relationship(dep=dep, ind=ind)
    com = Relationship(dep=dep + dep_shift + 0.5 * (slope - 1) * ind, ind=ind)
    com = ref.compute_limits(com)
    ref.build_response()
    com.build_response()
    current = -np.log(ref.score_response(com))
    rr, rc = ref._response_mean, com._response_mean
    d = rc - rr
    centered = np.sqrt(np.ma.mean((d - d.mean()) ** 2)) / np.sqrt(np.ma.mean((rr - rr.mean()) ** 2))
    return float(current), float(centered)


cur_b, cen_b = rel_errors(dep_shift=0.5)
cur_s, cen_s = rel_errors(slope=1.2)
report(
    "12 Relationship: a pure bias moves the current error; centering removes it",
    cur_b > 0.05 and cen_b < 1e-12 and cen_s > 0.05,
    f"bias of 0.5 in the dependent variable: current error {cur_b:.3f}, centered error "
    f"{cen_b:.1e}; slope 20% too steep: centered error {cen_s:.3f}",
)


# ---------------------------------------------------------------------------------
# Fix 14 (roll-up): a derived variable repeats the errors of its inputs.
# ---------------------------------------------------------------------------------
rng = np.random.default_rng(14)
gpp, reco = rng.uniform(2, 8, 1000), rng.uniform(1, 6, 1000)
eg, er = rng.normal(0, 1, 1000), rng.normal(0, 0.5, 1000)
nee_err = (reco + er - (gpp + eg)) - (reco - gpp)
identity = np.isclose(
    np.mean(nee_err**2), np.mean(er**2) + np.mean(eg**2) - 2 * np.mean(er * eg)
)
only_gpp = (reco - (gpp + eg)) - (reco - gpp)
twice = np.isclose(np.mean(only_gpp**2), np.mean(eg**2))
report(
    "14 roll-up: NEE repeats the GPP and respiration errors",
    identity and twice,
    f"MSE(NEE) = MSE(Reco) + MSE(GPP) - 2 cov: {identity}; a GPP-only error gives NEE the "
    f"same MSE as GPP, so a sum over GPP, Reco, and NEE counts it twice: {twice}",
)


if __name__ == "__main__":
    width = max(len(n) for n, _, _ in RESULTS)
    for name, passed, detail in RESULTS:
        print(f"{'PASS' if passed else 'FAIL'}  {name:<{width}}  {detail}")
