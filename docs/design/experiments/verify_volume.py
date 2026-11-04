"""
Checks for the volume adjustment of component scores (docs/design/probabilistic-framework.md).

Reuses the synthetic data of verify_components.py, which must be in the same
directory. Run from an ilamb3 checkout: python verify_volume.py
"""

import warnings

warnings.simplefilter("ignore")

import numpy as np
from verify_components import (
    AREA,
    TAU,
    R,
    anomaly,
    b,
    clim,
    e,
    ip,
    pattern,
    seasonal,
    smean,
    sm,
    sr,
    tmean,
    x,
)

RESULTS = []


def report(name, passed, detail):
    RESULTS.append((name, bool(passed), detail))


# -----------------------------------------------------------------------------------
# O. The reference's variance splits over the same subspaces as the error, so the
#    normalized MSE is the mean-bias term plus the volume-weighted mean of the
#    adjusted component errors.
# -----------------------------------------------------------------------------------
xt = x - smean(tmean(x))  # the reference about its overall mean (normalizer D_total)
x_pat = tmean(x) - smean(tmean(x)) + 0 * x
x_seas = clim(x - tmean(x))
x_anom = x - tmean(x) - x_seas
D = ip(xt, xt)
vol = {"pattern": ip(x_pat, x_pat), "seasonal": ip(x_seas, x_seas), "anomaly": ip(x_anom, x_anom)}
err = {"pattern": ip(pattern, pattern), "seasonal": ip(seasonal, seasonal), "anomaly": ip(anomaly, anomaly)}
nmse = ip(e, e) / D
E0 = float(smean(b)) ** 2 / D
v = {k: vol[k] / D for k in vol}
E2 = {k: err[k] / vol[k] for k in vol}
recon = E0 + sum(v[k] * E2[k] for k in vol)
taylor = (sm / sr) ** 2 + 1 - 2 * (sm / sr) * R
report(
    "O  NMSE = mean-bias term + volume-weighted mean of adjusted errors",
    np.isclose(sum(vol.values()), D) and np.isclose(recon, nmse) and np.isclose(E2["pattern"], taylor),
    f"volumes: pattern {v['pattern']:.1%}, seasonal {v['seasonal']:.1%}, anomaly "
    f"{v['anomaly']:.1%} of D; unadjusted shares "
    + ", ".join(f"{k} {err[k] / D:.3f}" for k in vol)
    + "; adjusted errors "
    + ", ".join(f"{k} {E2[k]:.3f}" for k in vol)
    + f"; mean bias {E0:.3f}; reconstructed NMSE {recon:.4f} = {nmse:.4f}; adjusted "
    f"pattern error = s_hat^2 + 1 - 2 s_hat R: {np.isclose(E2['pattern'], taylor)}",
)


# -----------------------------------------------------------------------------------
# P. Many components: splitting the centered error by cell (648 components, as in the
#    score maps) or by cell and calendar month (7776 components). Unadjusted scores
#    approach 1; adjusted scores stay on the scale of the total.
# -----------------------------------------------------------------------------------
c = e - b
xc = x - tmean(x)  # the reference about each cell's time mean (normalizer D_time)
total_E = np.sqrt(ip(c, c) / ip(xc, xc))


def split(group):
    """Squared error and reference volume of each component, as flat arrays."""
    w_all = (AREA * TAU).broadcast_like(c)
    if group == "cell":
        num = (c**2 * w_all).sum("time").values.ravel()
        den = (xc**2 * w_all).sum("time").values.ravel()
    else:
        num = (c**2 * w_all).groupby("time.month").sum("time").values.ravel()
        den = (xc**2 * w_all).groupby("time.month").sum("time").values.ravel()
    return num, den


rows = []
ok = True
for group in ("cell", "cell and calendar month"):
    num, den = split(group)
    N_k = num / den.sum()  # share of the normalized error (unadjusted)
    E_k = np.sqrt(num / den)  # adjusted
    v_k = den / den.sum()
    ok &= np.isclose(np.sum(v_k * E_k**2), total_E**2)
    rows.append(
        f"by {group} ({num.size} components): median unadjusted score "
        f"{np.median(np.exp(-np.sqrt(N_k))):.3f}, median adjusted score "
        f"{np.median(np.exp(-E_k)):.3f}"
    )
local_map = np.sqrt(tmean(c**2) / tmean(xc**2)).values.ravel()
num, den = split("cell")
ok &= np.allclose(np.sqrt(num / den), local_map)
report(
    "P  many components: adjusted scores stay on the scale of the total",
    ok,
    f"total score {np.exp(-total_E):.3f}; "
    + "; ".join(rows)
    + "; volume-weighted mean of adjusted squared errors = total; per-cell adjusted error "
    "= the RMSE score map's local error",
)

# -----------------------------------------------------------------------------------
# Q. With s = exp(-alpha NMSE), the total score is the mean-bias score times the
#    volume-weighted geometric mean of the adjusted component scores.
# -----------------------------------------------------------------------------------
alpha = 1.0
s_total = np.exp(-alpha * nmse)
s0 = np.exp(-alpha * E0)
s_k = {k: np.exp(-alpha * E2[k]) for k in vol}
product = s0 * np.prod([s_k[k] ** v[k] for k in vol])
report(
    "Q  exp(-alpha NMSE): total = mean-bias score x weighted geometric mean",
    np.isclose(s_total, product),
    f"total {s_total:.4f}; mean-bias score {s0:.4f}; adjusted component scores "
    + ", ".join(f"{k} {s_k[k]:.4f}" for k in vol)
    + f"; product {product:.4f}",
)

# -----------------------------------------------------------------------------------
# R. The thought experiment: 1000 components with equal volumes. Equal error shares
#    give every component the total's score; concentrated error does not.
# -----------------------------------------------------------------------------------
K, nmse_t = 1000, 0.5
vols = np.full(K, 1 / K)
even = np.full(K, nmse_t / K)
conc = np.zeros(K)
conc[0] = nmse_t
for name, shares in (("even", even), ("concentrated", conc)):
    adj = shares / vols
    assert np.isclose(np.sum(vols * adj), nmse_t)
even_adj, conc_adj = even / vols, conc / vols
report(
    "R  1000 equal volumes",
    np.allclose(even_adj, nmse_t) and np.isclose(conc_adj[0], nmse_t * K),
    f"total NMSE {nmse_t}: even error gives every component an unadjusted score "
    f"{np.exp(-np.sqrt(even[0])):.3f} and an adjusted score {np.exp(-np.sqrt(even_adj[0])):.3f} "
    f"(the total's {np.exp(-np.sqrt(nmse_t)):.3f}); error in one component gives it an "
    f"adjusted NMSE of {conc_adj[0]:.0f} and the other 999 a score of 1",
)

if __name__ == "__main__":
    width = max(len(name) for name, _, _ in RESULTS)
    for name, passed, detail in RESULTS:
        print(f"{'PASS' if passed else 'FAIL'}  {name:<{width}}  {detail}")
