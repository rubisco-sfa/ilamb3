"""
Checks for the general rule for component volumes (docs/design/probabilistic-framework.md).

A daily series of n = 356 days with equal weights, decomposed in time, in frequency,
and along a random orthogonal split. Needs only numpy: python verify_weights.py
"""

import numpy as np

rng = np.random.default_rng(11)
n = 356
t = np.arange(n)
RESULTS = []


def report(name, passed, detail):
    RESULTS.append((name, bool(passed), detail))


def ar1(phi, size):
    z = np.zeros(size)
    z[0] = rng.normal()
    eps = rng.normal(0, np.sqrt(1 - phi**2), size)
    for i in range(1, size):
        z[i] = phi * z[i - 1] + eps[i]
    return z


PHI = 0.9
x = 10 + 3 * np.sin(2 * np.pi * t / n) + ar1(PHI, n)
f = x + 0.4 + 0.8 * np.sin(2 * np.pi * t / n + 0.5) + 0.6 * ar1(0.5, n)
e = f - x
xt = x - x.mean()
D = xt @ xt
NMSE = (e @ e) / D


# Orthogonal projections (equal weights, so orthogonal means symmetric and idempotent)
def window(idx):
    p = np.zeros((n, n))
    p[idx, idx] = 1.0
    return p


def band(lo, hi):
    """Projection onto the real Fourier modes with rfft indices lo..hi-1."""
    mask = np.zeros(n // 2 + 1)
    mask[lo:hi] = 1.0
    return np.fft.irfft(np.fft.rfft(np.eye(n), axis=0) * mask[:, None], n=n, axis=0)


q, _ = np.linalg.qr(rng.normal(size=(n, n)))
DECOMP = {
    "four seasons": [window(np.arange(89 * k, 89 * (k + 1))) for k in range(4)],
    "355 days + 1 day": [window(np.arange(355)), window(np.array([355]))],
    "mean, low, high frequency": [band(0, 1), band(1, 19), band(19, n // 2 + 1)],
    "random 3-way": [q[:, a:b] @ q[:, a:b].T for a, b in ((0, 100), (100, 200), (200, n))],
}
for parts in DECOMP.values():
    total = sum(parts)
    assert np.allclose(total, np.eye(n), atol=1e-10)
    for p in parts:
        assert np.allclose(p, p.T, atol=1e-10) and np.allclose(p @ p, p, atol=1e-9)

# Baselines: covariance operators rho with trace D; volumes are tr(rho P)
lag = np.abs(t[:, None] - t[None, :])
a = rng.normal(size=(n, n))
BASE = {
    "white": np.eye(n) * D / n,
    "realized reference": np.outer(xt, xt),
    "expected AR(1)": PHI**lag * D / n,
    "random": a @ a.T * D / np.trace(a @ a.T),
}


def adjusted(rho, parts):
    """Volumes, adjusted errors, and the recovered NMSE (zero volume: normalize by D)."""
    vols, errs, recon = [], [], 0.0
    for p in parts:
        vol, err = float(np.trace(rho @ p)), float(e @ p @ e)
        vols.append(vol)
        if vol > 1e-9 * D:
            errs.append(err / vol)
            recon += vol / D * errs[-1]
        else:
            errs.append(np.nan)
            recon += err / D
    return np.array(vols) / D, np.array(errs), recon


# -----------------------------------------------------------------------------------
# S. Any baseline recovers the NMSE from any orthogonal decomposition.
# -----------------------------------------------------------------------------------
ok = all(
    np.isclose(adjusted(rho, parts)[2], NMSE)
    for rho in BASE.values()
    for parts in DECOMP.values()
)
report(
    "S  identity for every baseline and decomposition",
    ok,
    f"NMSE {NMSE:.4f} recovered for {len(BASE)} baselines x {len(DECOMP)} "
    "decompositions (time windows, frequency bands, random)",
)

# -----------------------------------------------------------------------------------
# T. Refinement: splitting one component changes nothing else. Equal weights per
#    component (1/2 for bias and variance) fail.
# -----------------------------------------------------------------------------------
coarse = [band(0, 1), band(1, n // 2 + 1)]
fine = [band(0, 1), band(1, 19), band(19, n // 2 + 1)]
ok = True
for rho in BASE.values():
    v_c, e_c, _ = adjusted(rho, coarse)
    v_f, e_f, _ = adjusted(rho, fine)
    ok &= np.isclose(v_c[0], v_f[0]) and (np.isnan(e_c[0]) or np.isclose(e_c[0], e_f[0]))
    ok &= np.isclose(e_c[1], (v_f[1] * e_f[1] + v_f[2] * e_f[2]) / v_c[1])
bias_half = float(e @ coarse[0] @ e) / (D / 2)
bias_third = float(e @ fine[0] @ e) / (D / 3)
report(
    "T  refinement: volumes are additive; equal weights per component are not",
    ok and not np.isclose(bias_half, bias_third),
    "for every baseline, splitting the variance term into two bands leaves the bias "
    "term's volume and adjusted error unchanged, and the variance term's adjusted error "
    "is the volume-weighted mean of its parts; with equal weights per component, the "
    f"same split changes the bias term's adjusted error from {bias_half:.4f} to "
    f"{bias_third:.4f}",
)

# -----------------------------------------------------------------------------------
# U. Time windows: white and expected (stationary) baselines give the measure fractions,
#    355/356, 1/356, and 1/4. A single realization fluctuates around them.
# -----------------------------------------------------------------------------------
vw_d = adjusted(BASE["white"], DECOMP["355 days + 1 day"])[0]
ve_d = adjusted(BASE["expected AR(1)"], DECOMP["355 days + 1 day"])[0]
vr_d = adjusted(BASE["realized reference"], DECOMP["355 days + 1 day"])[0]
vw_s = adjusted(BASE["white"], DECOMP["four seasons"])[0]
vr_s = adjusted(BASE["realized reference"], DECOMP["four seasons"])[0]
p_k = np.array([89 / n] * 4)
s2 = np.array([np.mean(xt[89 * k : 89 * (k + 1)] ** 2) for k in range(4)])
formula = p_k * s2 / np.sum(p_k * s2)
report(
    "U  time windows: measure fractions, and p_k s_k^2 / sum p_j s_j^2",
    np.allclose(vw_d, [355 / 356, 1 / 356])
    and np.allclose(ve_d, [355 / 356, 1 / 356])
    and np.allclose(vw_s, 0.25)
    and np.allclose(vr_s, formula),
    f"white and expected baselines: {vw_d[0]:.5f} and {vw_d[1]:.5f} (355/356, 1/356); "
    f"seasons 1/4 each; realized reference: the single day holds {vr_d[1]:.5f} and the "
    "seasons " + ", ".join(f"{v:.3f}" for v in vr_s) + ", equal to p_k s_k^2 / sum p_j s_j^2",
)

# -----------------------------------------------------------------------------------
# V. Frequency: the volume of a band is its share of the baseline's spectrum, which is
#    its share of bandwidth only for a white baseline. The mean is the zero-frequency
#    band, with volume 1/n_eff under the expected baseline.
# -----------------------------------------------------------------------------------
parts = DECOMP["mean, low, high frequency"]
vw = adjusted(BASE["white"], parts)[0]
ve = adjusted(BASE["expected AR(1)"], parts)[0]
vr = np.abs(adjusted(BASE["realized reference"], parts)[0])
var_mean = np.ones(n) @ (PHI**lag) @ np.ones(n) / n**2  # Var(mean) for unit variance
n_eff = 1 / var_mean
report(
    "V  frequency bands: spectral shares, not bandwidth shares",
    np.isclose(ve[0], 1 / n_eff) and np.isclose(vw[0], 1 / n) and np.isclose(vr[0], 0, atol=1e-12),
    f"mean / low (10% of frequencies) / high: white {vw[0]:.4f} / {vw[1]:.3f} / {vw[2]:.3f}; "
    f"expected AR(1) {ve[0]:.4f} / {ve[1]:.3f} / {ve[2]:.3f}; realized reference "
    f"{vr[0]:.4f} / {vr[1]:.3f} / {vr[2]:.3f}; mean volume 1/n_eff with n_eff = "
    f"{n_eff:.1f} for phi = {PHI}",
)

# -----------------------------------------------------------------------------------
# W. Other decompositions under the same rule: deciles of the reference (a partition
#    defined by the reference's values) and EOFs of the baseline (whose volumes are the
#    explained-variance fractions).
# -----------------------------------------------------------------------------------
dec = np.clip(np.searchsorted(np.quantile(x, np.linspace(0, 1, 11)), x, side="right") - 1, 0, 9)
deciles = [window(np.flatnonzero(dec == k)) for k in range(10)]
vw_q = adjusted(BASE["white"], deciles)[0]
ve_q = adjusted(BASE["expected AR(1)"], deciles)[0]
vr_q = adjusted(BASE["realized reference"], deciles)[0]
lam, u = np.linalg.eigh(BASE["random"])
lam, u = lam[::-1], u[:, ::-1]
eofs = [u[:, a:b] @ u[:, a:b].T for a, b in ((0, 5), (5, 50), (50, n))]
v_eof, _, recon_eof = adjusted(BASE["random"], eofs)
expl = np.array([lam[:5].sum(), lam[5:50].sum(), lam[50:].sum()]) / lam.sum()
report(
    "W  quantile bins and EOFs",
    np.allclose(vw_q, [np.mean(dec == k) for k in range(10)])
    and np.allclose(ve_q, vw_q)
    and np.isclose(vr_q.sum(), 1)
    and np.allclose(v_eof, expl)
    and np.isclose(recon_eof, NMSE),
    "deciles of the reference: white and expected volumes = the share of days in each decile "
    f"({vw_q.min():.3f}-{vw_q.max():.3f}); realized-reference volumes lowest / middle two "
    f"/ highest decile {vr_q[0]:.3f} / {vr_q[4]:.3f}, {vr_q[5]:.3f} / {vr_q[9]:.3f}; "
    "EOF groups of a baseline (1-5, 6-50, rest): volumes = explained-variance fractions "
    + " / ".join(f"{v_:.3f}" for v_ in v_eof),
)

if __name__ == "__main__":
    width = max(len(name) for name, _, _ in RESULTS)
    for name, passed, detail in RESULTS:
        print(f"{'PASS' if passed else 'FAIL'}  {name:<{width}}  {detail}")
