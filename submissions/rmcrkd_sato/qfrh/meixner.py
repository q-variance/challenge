"""Meixner (delta = 1/2) Sato process: the additive process with exact q-variance.

For a process with independent increments, E[QV_T e^{ipX_T}] = -Psi_T''(p) phi_T(p),
with Psi_T = log phi_T. Imposing E[QV_T | X_T = x] = s0^2 T + (x - x0)^2 / 2 for all x
gives the Riccati equation (Psi' - i x0)' = (Psi' - i x0)^2 - a^2, a^2 = 2 s0^2 T, whose
unique solution is the Meixner law with delta = 1/2:

    phi_T(p) = cos(beta/2) / cosh(a p - i beta/2) * exp(i x0 p),
    a = s0 sqrt(2T),  x0 = z0 sqrt(T),  beta = -2 arctan(z0 / (s0 sqrt 2)).

The family scales as sqrt(T) at fixed beta, so it is a Sato process of index 1/2.
Time is in years; s0 is the annualised minimum volatility, z0 the offset in z units.
"""
import numpy as np

DAYS = 252


def params(s0, z0):
    return s0 * np.sqrt(2.0), -2.0 * np.arctan(z0 / (s0 * np.sqrt(2.0)))


def cumulant(p, t, s0, z0):
    """Psi_t(p) = log E exp(i p X_t), and its first two p-derivatives."""
    a1, beta = params(s0, z0)
    a = a1 * np.sqrt(t)
    w = a * p - 0.5j * beta
    x0 = z0 * np.sqrt(t)
    psi = np.log(np.cos(beta / 2)) - np.log(np.cosh(w)) + 1j * x0 * p
    d1 = -a * np.tanh(w) + 1j * x0
    d2 = -(a / np.cosh(w)) ** 2
    return psi, d1, d2


def log_returns(n_paths, n_days, s0, z0, K=24, t0=0.0, rng=None):
    """Exact daily increments of the Meixner-Sato process started at age t0 (years).

    Uses cosh(w) = prod_k (1 + 4 w^2 / ((2k-1) pi)^2): each factor of the increment
    characteristic function is a pair of zero-inflated exponentials (probability of
    zero sqrt(s/t)), one negative with mean b/(1+kappa) and one positive with mean
    b/(1-kappa), b = 2 a_t / ((2k-1) pi), kappa = beta / ((2k-1) pi). Factors beyond K
    are replaced by a Gaussian with the exact remaining variance. Returns are centred.
    """
    rng = np.random.default_rng(rng)
    a1, beta = params(s0, z0)
    k = np.arange(1, K + 1)
    kap = beta / ((2 * k - 1) * np.pi)
    x = np.empty((n_paths, n_days))
    for i in range(n_days):
        s, t = t0 + i / DAYS, t0 + (i + 1) / DAYS
        r = s / t
        b = 2 * a1 * np.sqrt(t) / ((2 * k - 1) * np.pi)
        m_neg, m_pos = b / (1 + kap), b / (1 - kap)
        q = 1 - np.sqrt(r)  # probability each exponential is present
        neg = (rng.random((n_paths, K)) < q) * rng.exponential(m_neg, (n_paths, K))
        pos = (rng.random((n_paths, K)) < q) * rng.exponential(m_pos, (n_paths, K))
        mean_k = q * (m_pos - m_neg)
        var_k = (1 - r) * (m_neg ** 2 + m_pos ** 2)
        var_tot = 2 * s0 ** 2 * (t - s) * _var_factor(beta)
        rem = max(var_tot - var_k.sum(), 0.0)
        x[:, i] = (pos - neg - mean_k).sum(axis=1) + np.sqrt(rem) * rng.standard_normal(n_paths)
    return x


def _var_factor(beta):
    """Var(X_t) / (2 s0^2 t) = 1 / cos^2(beta/2) (from -Psi''(0) = a^2 sech^2(-i beta/2))."""
    return 1.0 / np.cos(beta / 2) ** 2


def conditional_rv(s0, z0, T, zgrid, n_steps=None, L=40.0, n=2 ** 14):
    """Semi-analytic E[annualised variance | z] for an anchored window of T days.

    n_steps=None gives continuous monitoring (quadratic variation). Otherwise the
    challenge estimator: n_steps equal sub-periods, np.std(ddof=0) of the returns.
    Uses E[sum r_i^2 e^{ipX}] = phi(p) sum_i (-Psi_i'' - Psi_i'^2), where Psi_i is the
    cumulant of the i-th increment, and sum (r_i - rbar)^2 = sum r_i^2 - X^2 / N.
    """
    tT = T / DAYS
    dx = 2 * L / n
    xs = (np.arange(n) - n // 2) * dx * np.sqrt(tT)          # x grid (log return)
    p = 2 * np.pi * np.fft.fftfreq(n, d=dx * np.sqrt(tT))
    psi, d1, d2 = cumulant(p, tT, s0, z0)
    phi = np.exp(psi)
    if n_steps is None:
        g = -d2 * phi
    else:
        ts = np.linspace(0, tT, n_steps + 1)
        acc = 0
        prev = [np.zeros_like(p, dtype=complex)] * 3
        for t in ts[1:]:
            cur = cumulant(p, t, s0, z0)
            e1, e2 = cur[1] - prev[1], cur[2] - prev[2]
            acc = acc + (-e2 - e1 ** 2)
            prev = cur
        g = acc * phi

    def inv(h):  # f(x_k) = (1/2pi) int e^{-ipx} h(p) dp on the centred x grid
        return np.real(np.fft.fftshift(np.fft.fft(h))) / (n * dx * np.sqrt(tT))

    f = inv(phi)
    num = inv(g)
    cond = num / f                                         # E[sum r^2 | x] (years units)
    if n_steps is not None:
        cond = cond - xs ** 2 / n_steps
    var_ann = cond / tT                                    # per-year variance
    z = xs / np.sqrt(tT)
    # mean of X is zero, so z needs no de-meaning
    return np.interp(zgrid, z, var_ann), np.interp(zgrid, z, f * np.sqrt(tT))

