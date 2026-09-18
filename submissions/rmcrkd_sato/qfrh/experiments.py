"""Dataset builders for the FRH experiments."""
import numpy as np
import pandas as pd
from .model import simulate_log_returns, gamma_fn
from . import meixner
from .qvar import window_stats, finalise, HORIZONS


def anchored_dataset(sigma, rho, gamma, days=1_000_000, horizons=HORIZONS, seed=0,
                     chunk=200_000, **kw):
    sim = lambda m, T, rng: simulate_log_returns(m, T, sigma, rho, gamma, rng=rng, **kw)
    return anchored(sim, days, horizons, seed, chunk)


def anchored_meixner(s0, z0, days=1_000_000, horizons=HORIZONS, seed=0, chunk=100_000, **kw):
    sim = lambda m, T, rng: meixner.log_returns(m, T, s0, z0, rng=rng, **kw)
    return anchored(sim, days, horizons, seed, chunk)


def anchored(sim, days, horizons, seed, chunk):
    """Forward-looking q-variance: every window of length T starts at the process origin.

    For each T, days // T independent windows are simulated, so the data volume
    per T matches a single series of `days` days cut as in data_loader_csv.py.
    """
    rng = np.random.default_rng(seed)
    out = []
    for T in horizons:
        n = days // T
        for k in range(0, n, chunk):
            m = min(chunk, n - k)
            r = sim(m, T, rng)
            out.append(window_stats(r, T))
    return finalise(pd.concat(out, ignore_index=True))


def block_series(sim, n_days, block, seed=0, chunk_blocks=2000):
    """Single daily log-return series made of independent anchored paths of `block` days.

    This is the only way to embed an additive (non-stationary) model in one price
    series: the Sato clock restarts every `block` days. It is cyclo-stationary, and
    `block` is an extra (structural) parameter.
    """
    rng = np.random.default_rng(seed)
    nb = -(-n_days // block)
    out = []
    for k in range(0, nb, chunk_blocks):
        m = min(chunk_blocks, nb - k)
        out.append(sim(m, block, rng).ravel())
    return np.concatenate(out)[:n_days]
