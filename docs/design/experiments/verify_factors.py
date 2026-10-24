"""
Checks for the adjustment factors 1/v_k (docs/design/probabilistic-framework.md).

Error spread evenly (white noise, the homogeneous baseline) over a daily series of
n = 356 days. Needs only numpy: python verify_factors.py
"""

import numpy as np

rng = np.random.default_rng(3)
n, draws = 356, 20000
RESULTS = []


def report(name, passed, detail):
    RESULTS.append((name, bool(passed), detail))


e = rng.normal(size=(draws, n))  # error spread evenly over all n degrees of freedom
total = np.sum(e**2, axis=1)  # D cancels in every ratio below, so use D = 1


def ratio(component, factor):
    """Mean adjusted error of a component over the mean total."""
    return np.mean(factor * component) / np.mean(total)


# -----------------------------------------------------------------------------------
# X. Four equal seasons: each holds about 1/4 of an evenly spread error, so its share
#    understates the total by a factor of 4; multiplying by 4 = 1/v restores it.
# -----------------------------------------------------------------------------------
seasons = [np.sum(e[:, 89 * k : 89 * (k + 1)] ** 2, axis=1) for k in range(4)]
share = [ratio(s, 1) for s in seasons]
adj = [ratio(s, 4) for s in seasons]
report(
    "X  seasons: factor 4 restores the total",
    np.allclose(share, 0.25, atol=0.01) and np.allclose(adj, 1, atol=0.01),
    "mean share / total: " + ", ".join(f"{v:.3f}" for v in share)
    + "; after multiplying by 4: " + ", ".join(f"{v:.3f}" for v in adj),
)

# -----------------------------------------------------------------------------------
# Y. Bias and variance: an evenly spread error puts 1/n of itself in the mean, so the
#    factors are n and n/(n-1), not 2 and 2.
# -----------------------------------------------------------------------------------
bias = n * e.mean(axis=1) ** 2
var = np.sum((e - e.mean(axis=1, keepdims=True)) ** 2, axis=1)
assert np.allclose(bias + var, total)
two = (ratio(bias, 2), ratio(var, 2))
vol = (ratio(bias, n), ratio(var, n / (n - 1)))
report(
    "Y  bias/variance: factors n and n/(n-1), not 2 and 2",
    np.allclose(vol, 1, atol=0.03) and two[0] < 0.01 and two[1] > 1.9,
    f"mean share / total: bias {ratio(bias, 1):.4f} (1/n = {1 / n:.4f}), variance "
    f"{ratio(var, 1):.4f}; factors 2 and 2 give bias {two[0]:.4f} and variance "
    f"{two[1]:.3f}; factors {n} and {n / (n - 1):.4f} give bias {vol[0]:.3f} and "
    f"variance {vol[1]:.3f}",
)

# -----------------------------------------------------------------------------------
# Z. The volumes are the coefficients of the average that gives back the total, so
#    they sum to 1; the factors are their reciprocals. One draw, exactly.
# -----------------------------------------------------------------------------------
i = 0
seas_adj = [4 * s[i] for s in seasons]
bv_adj = (n * bias[i], n / (n - 1) * var[i])
report(
    "Z  volume-weighted mean of adjusted errors = total",
    np.isclose(np.sum([0.25 * a for a in seas_adj]), total[i])
    and np.isclose(bv_adj[0] / n + bv_adj[1] * (n - 1) / n, total[i]),
    f"one draw: total {total[i]:.2f}; seasons sum(1/4 x adjusted) "
    f"{np.sum([0.25 * a for a in seas_adj]):.2f}; bias/variance "
    f"(1/n) x {bv_adj[0]:.2f} + ((n-1)/n) x {bv_adj[1]:.2f} = "
    f"{bv_adj[0] / n + bv_adj[1] * (n - 1) / n:.2f}",
)

# -----------------------------------------------------------------------------------
# AA. Consistency in aggregation: split the variance term into a low and a high
#     frequency band. With additive volumes, the variance term's adjusted error is the
#     average of its bands' adjusted errors, with coefficients that sum to 1. With equal
#     volumes per component (1/2, then 1/3), the coefficients that reproduce it for every
#     error sum to 4/3, so it is not an average.
# -----------------------------------------------------------------------------------
def band(lo, hi):
    mask = np.zeros(n // 2 + 1)
    mask[lo:hi] = 1.0
    return np.fft.irfft(np.fft.rfft(np.eye(n), axis=0) * mask[:, None], n=n, axis=0)


low, high = band(1, 19), band(19, n // 2 + 1)
ee = e[:2]  # two draws give two equations for the two coefficients
s_lo = np.einsum("di,ij,dj->d", ee, low, ee)
s_hi = np.einsum("di,ij,dj->d", ee, high, ee)
s_var = var[:2]
assert np.allclose(s_lo + s_hi, s_var)
v_lo, v_hi = np.trace(low) / n, np.trace(high) / n  # 36/n and 319/n
v_var = (n - 1) / n
coef_add = np.linalg.solve(np.column_stack([s_lo / v_lo, s_hi / v_hi]), s_var / v_var)
coef_eq = np.linalg.solve(np.column_stack([3 * s_lo, 3 * s_hi]), 2 * s_var)
report(
    "AA consistency in aggregation requires additive volumes",
    np.isclose(v_lo + v_hi, v_var)
    and np.allclose(coef_add, [v_lo / v_var, v_hi / v_var])
    and np.isclose(coef_add.sum(), 1)
    and np.allclose(coef_eq, 2 / 3),
    f"additive volumes (variance {v_var:.4f} = low {v_lo:.4f} + high {v_hi:.4f}): "
    f"coefficients {coef_add[0]:.4f} + {coef_add[1]:.4f} = {coef_add.sum():.4f}; equal "
    f"volumes per component (1/2, then 1/3 each): coefficients {coef_eq[0]:.4f} + "
    f"{coef_eq[1]:.4f} = {coef_eq.sum():.4f}",
)

if __name__ == "__main__":
    width = max(len(name) for name, _, _ in RESULTS)
    for name, passed, detail in RESULTS:
        print(f"{'PASS' if passed else 'FAIL'}  {name:<{width}}  {detail}")
