"""
Checks for the choice of score function (docs/design/probabilistic-framework.md).

Reuses the synthetic data of verify_volume.py and verify_components.py, which must be
in the same directory. Run from an ilamb3 checkout: python verify_scores.py
"""

import warnings

warnings.simplefilter("ignore")

import numpy as np
import verify_volume as volume
from scipy import integrate, stats

from ilamb3.analysis import scoring

rng = np.random.default_rng(7)
RESULTS = []


def report(name, passed, detail):
    RESULTS.append((name, bool(passed), detail))


SCORES = {
    "exp(-ln2 NMSE)": lambda t: np.exp(-np.log(2) * t),
    "exp(-sqrt NMSE) (ilamb3)": scoring.score_nmse,
    "1/(1 + NMSE)": lambda t: 1 / (1 + t),
}

# -----------------------------------------------------------------------------------
# AB. Only the exponential of the NMSE turns the sum of components into a product of
#     scores. On the check O data, the total is the mean-bias score times the
#     volume-weighted geometric mean of the adjusted component scores.
# -----------------------------------------------------------------------------------
x, y = 0.7, 1.9
rows = [f"{k}: s(x+y) {s(x + y):.3f}, s(x)s(y) {s(x) * s(y):.3f}" for k, s in SCORES.items()]
E0, v, E2, nmse = volume.E0, volume.v, volume.E2, volume.nmse
prod_ok, totals = True, []
for alpha in (np.log(2), 1.0):
    total = np.exp(-alpha * nmse)
    product = np.exp(-alpha * E0) * np.prod([np.exp(-alpha * E2[k]) ** v[k] for k in v])
    prod_ok &= np.isclose(total, product)
    totals.append(f"alpha {alpha:.3f}: total {total:.4f} = product {product:.4f}")
report(
    "AB product rule",
    prod_ok
    and np.isclose(SCORES["exp(-ln2 NMSE)"](x + y), SCORES["exp(-ln2 NMSE)"](x) * SCORES["exp(-ln2 NMSE)"](y))
    and not np.isclose(SCORES["exp(-sqrt NMSE) (ilamb3)"](x + y), SCORES["exp(-sqrt NMSE) (ilamb3)"](x) * SCORES["exp(-sqrt NMSE) (ilamb3)"](y)),
    f"x = {x}, y = {y}; " + "; ".join(rows) + "; check O: " + "; ".join(totals),
)

# -----------------------------------------------------------------------------------
# AC. With Gaussian errors of variance c times the reference variance, the likelihood
#     ratio of the model to a perfect model is exp(-(n/2c) NMSE), and it factorizes
#     over orthogonal components, which are independent under that error model.
# -----------------------------------------------------------------------------------
n, c = 500, 250.0
ref = rng.normal(3, 2, n)
err = rng.normal(0.3, 1.1, n)
var_ref = np.mean((ref - ref.mean()) ** 2)
nmse_ac = np.mean(err**2) / var_ref
log_lr = -np.sum(err**2) / (2 * c * var_ref)
q, _ = np.linalg.qr(rng.normal(size=(n, n)))
P1 = q[:, :50] @ q[:, :50].T
P2 = np.eye(n) - P1
parts = [-(err @ P @ err) / (2 * c * var_ref) for P in (P1, P2)]
Z = rng.normal(size=(20000, n))
corr = np.corrcoef(np.sum((Z @ P1) * Z, axis=1), np.sum((Z @ P2) * Z, axis=1))[0, 1]
report(
    "AC Gaussian likelihood",
    np.isclose(np.exp(log_lr), np.exp(-(n / (2 * c)) * nmse_ac))
    and np.isclose(log_lr, sum(parts))
    and abs(corr) < 0.03,
    f"likelihood ratio {np.exp(log_lr):.4f} = exp(-(n/2c) NMSE) "
    f"{np.exp(-(n / (2 * c)) * nmse_ac):.4f} (n = {n}, c = {c:g}); the log-likelihood "
    f"is the sum over two orthogonal components; correlation of their squared norms "
    f"over 20000 isotropic draws {corr:.3f}",
)

# -----------------------------------------------------------------------------------
# AD. Mixtures of exponential scores over alpha (Bernstein's theorem): an Exp(1)
#     mixture is 1/(1 + x), a Gamma(1/p, scale p) mixture is (1 + p x)^(-1/p). Their
#     totals are the harmonic and power means of the component scores.
# -----------------------------------------------------------------------------------
p = 0.5
mix_ok, mix_rows = True, []
for xx in (0.25, 1.0, 4.0):
    m_exp = integrate.quad(lambda a: np.exp(-a * xx) * np.exp(-a), 0, np.inf)[0]
    m_gam = integrate.quad(lambda a: np.exp(-a * xx) * stats.gamma.pdf(a, 1 / p, scale=p), 0, np.inf)[0]
    mix_ok &= np.isclose(m_exp, 1 / (1 + xx)) and np.isclose(m_gam, (1 + p * xx) ** (-1 / p))
    mix_rows.append(f"x = {xx}: {m_exp:.4f} and {m_gam:.4f}")
vk = rng.dirichlet(np.ones(5))
Ek = rng.gamma(2, 0.5, 5)
harm = 1 / np.sum(vk / (1 / (1 + Ek)))
powm = np.sum(vk * ((1 + p * Ek) ** (-1 / p)) ** (-p)) ** (-1 / p)
report(
    "AD mixtures over alpha",
    mix_ok
    and np.isclose(1 / (1 + vk @ Ek), harm)
    and np.isclose((1 + p * (vk @ Ek)) ** (-1 / p), powm),
    "Exp(1) and Gamma(2, scale 0.5) mixtures of exp(-alpha x) equal 1/(1+x) and "
    "(1+x/2)^-2: " + "; ".join(mix_rows) + f"; for 5 random components, the total "
    f"1/(1 + sum v E) = {1 / (1 + vk @ Ek):.4f} = harmonic mean of component scores "
    f"{harm:.4f}, and the gamma mixture's total equals the power mean of order -{p}",
)

# -----------------------------------------------------------------------------------
# AE. Error intervals. The share of errors within b +/- 2 s_e (mean error +/- 2 x the
#     standard deviation of the error) is at least 3/4 for any distribution
#     (Chebyshev), at least 8/9 for a unimodal one (Vysochanskij-Petunin), and 0.954
#     for a normal one. The bounds hold exactly for the area- and time-weighted
#     distribution of a model's errors, which is itself a distribution.
# -----------------------------------------------------------------------------------
def coverage(dist):
    m, s = dist.mean(), dist.std()
    return dist.cdf(m + 2 * s) - dist.cdf(m - 2 * s)


unimodal = {
    "normal": stats.norm(),
    "Laplace": stats.laplace(),
    "gamma(2), skewed": stats.gamma(2),
    "lognormal(1), skewed": stats.lognorm(1),
    "Student t(3)": stats.t(3),
    "uniform": stats.uniform(),
}
cov_u = {k: coverage(d) for k, d in unimodal.items()}
# Chebyshev's extreme case: mass 1/8 at -2, 3/4 at 0, 1/8 at +2
pts, prob = np.array([-2.0, 0.0, 2.0]), np.array([1 / 8, 3 / 4, 1 / 8])
m_c = prob @ pts
s_c = np.sqrt(prob @ (pts - m_c) ** 2)
cheb = prob @ (np.abs(pts - m_c) < 2 * s_c)
e, AREA, TAU = volume.e, volume.AREA, volume.TAU
w = (AREA * TAU).broadcast_like(e).values.ravel()
ev = e.values.ravel()
b = np.sum(w * ev) / np.sum(w)
s_e = np.sqrt(np.sum(w * (ev - b) ** 2) / np.sum(w))
rmse = np.sqrt(np.sum(w * ev**2) / np.sum(w))
cov_model = np.sum(w * (np.abs(ev - b) < 2 * s_e)) / np.sum(w)
cov_rmse = np.sum(w * (np.abs(ev) < 2 * rmse)) / np.sum(w)
report(
    "AE error intervals",
    all(c_ >= 8 / 9 for c_ in cov_u.values())
    and abs(cov_u["normal"] - 0.9545) < 1e-3
    and np.isclose(s_c, 1) and np.isclose(cheb, 0.75)
    and cov_model >= 0.75
    and cov_rmse >= 0.75,
    "coverage of mean +/- 2 sd: "
    + ", ".join(f"{k} {c_:.3f}" for k, c_ in cov_u.items())
    + f" (all >= 8/9 = {8 / 9:.3f}); Chebyshev's extreme case {cheb:.3f}; the synthetic "
    f"model's errors (area x time weighted): b = {b:.3f}, s_e = {s_e:.3f}, RMSE "
    f"{rmse:.3f}; {cov_model:.3f} of the domain within b +/- 2 s_e and {cov_rmse:.3f} "
    "within +/- 2 RMSE",
)

# -----------------------------------------------------------------------------------
# AF. In physical units, the components' root-mean-square errors add in quadrature to
#     the RMSE that ilamb3 pools over time and area, so each component's percentage of
#     the total MSE is well defined.
# -----------------------------------------------------------------------------------
ip, b_map = volume.ip, volume.b
D = ip(volume.xt, volume.xt)
ms = {
    "mean bias": float(volume.smean(b_map)) ** 2,
    "pattern": ip(volume.pattern, volume.pattern),
    "seasonal": ip(volume.seasonal, volume.seasonal),
    "anomaly": ip(volume.anomaly, volume.anomaly),
}
total_ms = ip(e, e)
pooled = scoring.aggregate(scoring.aggregate(scoring.statistics(e), dim="time"))
rmse_ilamb3 = float(np.sqrt(pooled["mse"]))
report(
    "AF error budget in physical units",
    np.isclose(sum(ms.values()), total_ms) and np.isclose(np.sqrt(total_ms), rmse_ilamb3),
    f"ilamb3's RMSE {rmse_ilamb3:.3f} = sqrt of the sum of squares of "
    + ", ".join(f"{k} {np.sqrt(m):.3f}" for k, m in ms.items())
    + "; percentages of the total MSE: "
    + ", ".join(f"{k} {m / total_ms:.1%}" for k, m in ms.items())
    + f"; skill score 1 - NMSE = {1 - total_ms / D:.3f}",
)

if __name__ == "__main__":
    width = max(len(name) for name, _, _ in RESULTS)
    for name, passed, detail in RESULTS:
        print(f"{'PASS' if passed else 'FAIL'}  {name:<{width}}  {detail}")
