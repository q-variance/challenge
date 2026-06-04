#!/usr/bin/env python3
"""
Coherent inverse-chi-square Hermite-energy model for the Q-Variance Challenge.

This script generates a daily price path and writes a CSV with a single column
named 'Price', as expected by the official q-variance challenge loader.

Default calibrated parameters:
    beta_mult = 1.719800
    memory    = 15.275991
    eta       = 0.498969

The model has three exposed parameters: beta_mult, memory, eta.
"""

from __future__ import annotations

import argparse
import math
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import lfilter
    from scipy.stats import norm, chi2
except Exception as exc:  # pragma: no cover
    raise SystemExit("Install dependencies with: pip install numpy pandas scipy") from exc

# Benchmark annual volatility used as a numerical unit for the scale parameter.
# Equivalently, beta = beta_mult * SIGMA0**2 is the free annual variance scale.
SIGMA0 = 0.2586
TRADING_DAYS = 252.0
NU_FIXED = 3.0

DEFAULT_BETA_MULT = 1.719800
DEFAULT_MEMORY = 15.275991
DEFAULT_ETA = 0.498969
DEFAULT_SEED = 5


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - float(np.mean(x))
    s = float(np.std(x, ddof=0))
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("cannot standardize degenerate array")
    return x / s


def _ar1_from_noise(noise: np.ndarray, memory: float, x0: float = 0.0) -> np.ndarray:
    if memory <= 0.0:
        raise ValueError("memory must be positive")
    rho = math.exp(-1.0 / float(memory))
    rho = min(max(rho, 0.0), 0.9999995)
    sigma = math.sqrt(max(0.0, 1.0 - rho * rho))
    y, _ = lfilter([sigma], [1.0, -rho], np.asarray(noise, dtype=float), zi=np.array([rho * x0], dtype=float))
    return y


def ar1_gaussian(n: int, memory: float, rng: np.random.Generator) -> np.ndarray:
    # Burn-in proportional to memory so that the field is close to stationary.
    burn = int(min(max(1000, round(12.0 * memory)), 250000))
    noise = rng.standard_normal(n + burn)
    x0 = rng.standard_normal()
    y = _ar1_from_noise(noise, memory, x0=x0)
    return _standardize(y[burn:])


def persistent_inverse_chisq_activity(n: int, seed: int, memory: float) -> np.ndarray:
    """Persistent inverse-chi-square baseline activity.

    A Gaussian copula creates a persistent chi-square field.  The activity is
    the inverse chi-square radius with nu = 3, then normalized to mean one.
    """
    rng = np.random.default_rng(int(seed) + 424242)
    u = ar1_gaussian(n, memory, rng)
    p = norm.cdf(u)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    q = chi2.ppf(p, df=NU_FIXED)
    q = np.maximum(q, 1e-300)
    activity = (NU_FIXED - 2.0) / q
    return activity / float(np.mean(activity))


def hermite_energy_feedback(eps: np.ndarray, memory: float, eta: float) -> np.ndarray:
    """Persistent Hermite-energy feedback with no independent skew parameter.

    The even energy term is H2(eps)/sqrt(2).  The odd H1 tilt is tied to eta,
    alpha_eff = eta^2/sqrt(2), so there is no separate alpha parameter.
    """
    eps = np.asarray(eps, dtype=float)
    h2 = eps * eps - 1.0
    alpha_eff = eta * eta / math.sqrt(2.0)
    drive = h2 / math.sqrt(2.0) - alpha_eff * eps
    drive = _standardize(drive)
    f = _ar1_from_noise(drive, memory, x0=0.0)
    return _standardize(f)


def generate_returns(
    n: int,
    seed: int,
    beta_mult: float = DEFAULT_BETA_MULT,
    memory: float = DEFAULT_MEMORY,
    eta: float = DEFAULT_ETA,
) -> np.ndarray:
    if beta_mult <= 0.0:
        raise ValueError("beta_mult must be positive")
    if memory <= 0.0:
        raise ValueError("memory must be positive")
    if eta < 0.0:
        raise ValueError("eta must be nonnegative")

    rng = np.random.default_rng(int(seed) + 989898)
    eps = rng.standard_normal(n)

    base_activity = persistent_inverse_chisq_activity(n, seed, memory)
    feedback_field = hermite_energy_feedback(eps, memory, eta)

    log_mod = eta * feedback_field
    mod = np.exp(log_mod - np.max(log_mod))
    mod = mod / float(np.mean(mod))

    activity = base_activity * mod
    activity = activity / float(np.mean(activity))

    v_annual = float(beta_mult) * SIGMA0 * SIGMA0 * activity
    returns = np.sqrt(v_annual / TRADING_DAYS) * eps
    returns = returns - float(np.mean(returns))
    return returns


def make_prices(returns: np.ndarray, p0: float = 100.0) -> np.ndarray:
    """Convert log returns to a numerically safe positive price path."""
    logp = np.cumsum(np.asarray(returns, dtype=float))
    mid = 0.5 * (float(np.min(logp)) + float(np.max(logp)))
    return np.exp(math.log(p0) + logp - mid)


def price_safety_stats(returns: np.ndarray) -> dict:
    r = np.asarray(returns, dtype=float)
    logp = np.cumsum(r)
    return {
        "log_range": float(np.max(logp) - np.min(logp)),
        "max_abs_return": float(np.max(np.abs(r))),
        "q999_abs_return": float(np.quantile(np.abs(r), 0.999)),
        "q9999_abs_return": float(np.quantile(np.abs(r), 0.9999)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000, help="Number of daily returns to simulate")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    ap.add_argument("--beta-mult", type=float, default=DEFAULT_BETA_MULT)
    ap.add_argument("--memory", type=float, default=DEFAULT_MEMORY)
    ap.add_argument("--eta", type=float, default=DEFAULT_ETA)
    ap.add_argument("--out", default="variance_timeseries.csv")
    ap.add_argument("--summary-out", default="submission_summary.csv")
    args = ap.parse_args()

    returns = generate_returns(args.n, args.seed, args.beta_mult, args.memory, args.eta)
    prices = make_prices(returns)
    pd.DataFrame({"Price": prices}).to_csv(args.out, index=False)

    stats = {
        "n": args.n,
        "seed": args.seed,
        "beta_mult": args.beta_mult,
        "memory": args.memory,
        "eta": args.eta,
        **price_safety_stats(returns),
    }
    pd.DataFrame([stats]).to_csv(args.summary_out, index=False)
    print(f"Saved {args.out}")
    print(f"Saved {args.summary_out}")
    print(pd.DataFrame([stats]).to_string(index=False))


if __name__ == "__main__":
    main()
