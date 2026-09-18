"""Implied volatility smiles of additive models via the Lewis (2001) formula.

Normalised call (forward 1, log-strike k), as in frh_fx.frh.ft_price:
    C(k) = 1 - 1/(2 pi) e^{k/2} int Re[e^{-ikp} phi(p - i/2)] / (p^2 + 1/4) dp,
where phi is the risk-neutral characteristic function of X_t = log(F_t / F_0).
Risk-neutral laws are obtained from the (physical) centred models by the deterministic
drift that makes exp(X_t) a martingale; this is the only change of measure assumed.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def rn_cumulant(cum, t):
    """Psi^Q(p) = Psi(p) - i p Psi(-i), i.e. X^Q_t = X_t - log E exp(X_t)."""
    c = cum(np.array([-1j]), t)[0]
    return lambda p: cum(p, t) - 1j * p * c


def lewis_calls(k, psi, L=200.0, n=2 ** 14):
    p = np.linspace(0.0, L, n + 1)[1:] - L / n / 2  # midpoint rule on (0, L)
    dp = L / n
    phi = np.exp(psi(p - 0.5j))
    integ = np.real(np.exp(-1j * np.outer(k, p)) * phi) / (p ** 2 + 0.25)
    return 1 - np.exp(k / 2) / np.pi * integ.sum(axis=1) * dp  # integrand even in p


def bs_call(k, v):
    s = np.sqrt(v)
    d1 = -k / s + 0.5 * s
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d1 - s)


def implied_vols(k, t, prices):
    out = []
    for ki, ci in zip(k, prices):
        lo = max(1 - np.exp(ki), 0)
        ci = min(max(ci, lo + 1e-14), 1 - 1e-14)
        out.append(brentq(lambda s: bs_call(ki, s * s * t) - ci, 1e-6, 10.0))
    return np.array(out)


def smile(cum, t, k):
    """Implied vols at log-strikes k for maturity t (years); cum(p, t) -> Psi_t(p)."""
    psi = rn_cumulant(lambda p, tt: cum(p, tt), t)
    return implied_vols(k, t, lewis_calls(k, psi))


def meixner_cum(model):
    return lambda p, t: model.cumulant(p, t)[0]


def frh_cum(model):
    return lambda p, t: model.increment(p, 0.0, t)[0]


def deltas(iv, k, t):
    """frh-fx convention (uts.get_deltas with rho = 0): Delta = N(k / (iv sqrt t))."""
    return norm.cdf(k / (iv * np.sqrt(t)))
