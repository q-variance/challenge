#!/usr/bin/env python3
"""
cos2_phase_hermite_submission.py

Three-parameter phase-Hermite oscillator generator for the Q-Variance Challenge.

The model combines a compact latent phase state, a two-harmonic activity map,
and a Hermite-deformed Gaussian shock.

Exposed parameters:
    beta_mult   annual variance scale multiplier
    g           strength of the phase activity modulation
    eta         Hermite deformation strength and phase-response strength
"""

from __future__ import annotations

import argparse
import math
import numpy as np
import pandas as pd

SIGMA0 = 0.2586
TWO_PI = 2.0 * math.pi
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
HERMITE_RATIO = 0.10

DEFAULT_MODE = "cos2"
DEFAULT_BETA_MULT = 2.645106721716812
DEFAULT_G = 1.68520494798523
DEFAULT_ETA = 0.2341884778705525
DEFAULT_SEED = 1


def hermite_y(eps: np.ndarray, eta: float) -> np.ndarray:
    h2 = eps * eps - 1.0
    h3 = eps * eps * eps - 3.0 * eps
    raw = eps + eta * (h3 - HERMITE_RATIO * h2)
    raw_var = 1.0 + eta * eta * (6.0 + 2.0 * HERMITE_RATIO * HERMITE_RATIO)
    return raw / math.sqrt(raw_var)


def phase_activity(eps: np.ndarray, g: float, eta: float, seed: int, mode: str = DEFAULT_MODE) -> np.ndarray:
    n = len(eps)
    theta = np.empty(n, dtype=float)
    rng = np.random.default_rng(int(seed) + 24681357)
    th = rng.uniform(0.0, TWO_PI)

    for t in range(n):
        theta[t] = th
        th = (th + GOLDEN_ANGLE - eta * eps[t]) % TWO_PI

    if mode == "cos2":
        logA = g * (np.cos(theta) + 0.5 * np.cos(2.0 * theta))
    else:
        raise ValueError("this submission uses mode='cos2'")

    A = np.exp(logA - np.max(logA))
    A = A / float(np.mean(A))
    return A


def generate_returns(
    n: int,
    seed: int = DEFAULT_SEED,
    beta_mult: float = DEFAULT_BETA_MULT,
    g: float = DEFAULT_G,
    eta: float = DEFAULT_ETA,
    mode: str = DEFAULT_MODE,
) -> np.ndarray:
    if beta_mult <= 0.0:
        raise ValueError("beta_mult must be positive")
    if g < 0.0:
        raise ValueError("g must be nonnegative")
    if eta < 0.0:
        raise ValueError("eta must be nonnegative")
    if mode != "cos2":
        raise ValueError("this submission uses mode='cos2'")

    rng = np.random.default_rng(int(seed))
    eps = rng.standard_normal(int(n))

    y = hermite_y(eps, eta)
    A = phase_activity(eps, g, eta, int(seed), mode=mode)

    r = math.sqrt(beta_mult * SIGMA0 * SIGMA0 / 252.0) * np.sqrt(A) * y
    r = r - float(np.mean(r))
    return r


def make_prices(returns: np.ndarray, p0: float = 100.0) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    logp = np.cumsum(r)
    lo = float(np.min(logp))
    hi = float(np.max(logp))
    mid = 0.5 * (lo + hi)
    return np.exp(math.log(p0) + logp - mid)


def path_summary(returns: np.ndarray, seed: int, beta_mult: float, g: float, eta: float) -> dict:
    r = np.asarray(returns, dtype=float)
    logp = np.cumsum(r)
    return {
        "n": int(len(r)),
        "seed": int(seed),
        "beta_mult": float(beta_mult),
        "g": float(g),
        "eta": float(eta),
        "log_range": float(np.max(logp) - np.min(logp)),
        "max_abs_return": float(np.max(np.abs(r))),
        "q999_abs_return": float(np.quantile(np.abs(r), 0.999)),
        "q9999_abs_return": float(np.quantile(np.abs(r), 0.9999)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--mode", choices=["cos2"], default=DEFAULT_MODE)
    ap.add_argument("--beta-mult", type=float, default=DEFAULT_BETA_MULT)
    ap.add_argument("--g", type=float, default=DEFAULT_G)
    ap.add_argument("--eta", type=float, default=DEFAULT_ETA)
    ap.add_argument("--out", default="variance_timeseries.csv")
    ap.add_argument("--summary-out", default="submission_summary.csv")
    args = ap.parse_args()

    returns = generate_returns(args.n, args.seed, args.beta_mult, args.g, args.eta, args.mode)
    prices = make_prices(returns)

    pd.DataFrame({"Price": prices}).to_csv(args.out, index=False)
    pd.DataFrame([path_summary(returns, args.seed, args.beta_mult, args.g, args.eta)]).to_csv(args.summary_out, index=False)

    print(f"Saved {args.out}")
    print(f"Saved {args.summary_out}")


if __name__ == "__main__":
    main()
