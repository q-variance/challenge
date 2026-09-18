"""Fast-reversion Heston (FRH) / NIG additive process with time-dependent gamma.

Follows rmcrkd/frh-fx (Mechkov 2015 parameterisation). Over a step of length dt
with kurtosis parameter g, the log-price increment is

    dX = -1/2 s^2 v + (rho s / g) (v - dt) + s sqrt(1 - rho^2) sqrt(v) eps,

where v ~ IG(mean dt, variance g^2 dt) is the (normalised) integrated variance
increment and eps ~ N(0,1). This is the kappa -> infinity limit of Heston with
leverage rho, and E[exp(dX)] = 1 exactly for any g with 1 - g rho s > 0.

The "delta symmetry" setup of frh-fx/5-symmetry.ipynb is gamma(t) = c t^0.5,
which makes X a Sato process (self-similar additive process) of index 1/2,
up to the -1/2 s^2 V convexity term.
"""
import numpy as np

DAYS = 252


def gamma_fn(c, H=0.5):
    """gamma(t) = c t^H, t in years since the process origin."""
    return lambda t: c * np.power(np.maximum(t, 0.0), H)


def _substep_grid(n_days, substeps, first_day_levels):
    """Time grid (years) with `substeps` per day and a geometric refinement of day 1."""
    dt = 1.0 / DAYS
    grid = [0.0]
    # geometric refinement of the first day, where gamma(t) ~ sqrt(t) varies most
    geo = dt * 2.0 ** -np.arange(first_day_levels, 0, -1)
    grid += list(geo)
    grid += list(dt + np.arange(1, (n_days - 1) * substeps + 1) * dt / substeps) if n_days > 1 else []
    # make sure day boundaries are in the grid
    g = np.unique(np.concatenate([grid, np.arange(n_days + 1) * dt]))
    return g


def simulate_log_returns(n_paths, n_days, sigma, rho, gamma, t0=0.0, substeps=4,
                         first_day_levels=8, convexity=True, rng=None):
    """Daily log returns (n_paths, n_days) of the FRH additive process started at age t0.

    gamma : callable of time (years since origin) giving the NIG kurtosis parameter.
    convexity : if True, include the -1/2 s^2 v term so that exp(X) is a martingale
        (risk-neutral FRH, as in frh-fx). If False, X itself is a martingale, which
        is the exactly self-similar (Sato) version of the delta-symmetric model.
    """
    rng = np.random.default_rng(rng)
    dt_day = 1.0 / DAYS
    grid = t0 + _substep_grid(n_days, substeps, first_day_levels if t0 == 0.0 else 0)
    t_a, t_b = grid[:-1], grid[1:]
    h = t_b - t_a
    # gamma^2 averaged over the step: matches the variance of the IG increment exactly
    tt = np.linspace(0, 1, 9)[:, None]
    g2 = np.mean(gamma(t_a + tt * h) ** 2, axis=0)
    g = np.sqrt(g2)

    x = np.zeros((n_paths, n_days))
    day_idx = np.minimum(((t_b - t0) / dt_day - 1e-9).astype(int), n_days - 1)
    s2 = sigma ** 2 if convexity else 0.0
    for j in range(len(h)):
        if g[j] < 1e-10:
            dx = -0.5 * s2 * h[j] + sigma * np.sqrt(h[j]) * rng.standard_normal(n_paths)
        else:
            v = rng.wald(h[j], h[j] ** 2 / g2[j], size=n_paths)
            eps = rng.standard_normal(n_paths)
            dx = (-0.5 * s2 * v + rho * sigma / g[j] * (v - h[j])
                  + sigma * np.sqrt(1 - rho ** 2) * np.sqrt(v) * eps)
        x[:, day_idx[j]] += dx
    return x
