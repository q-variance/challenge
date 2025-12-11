"""
Regime Mixture Q-Variance Model for Q-Variance Challenge

This module implements a regime-switching variance model where the log-price
follows a piecewise-constant variance process with regime changes.

Model:
- Regime-switching variance: V_j = 1/τ_j where τ_j ~ Gamma(3/2, rate=σ₀²)
- Log-price increments: ΔL_t ~ N(μ*dt, V_j*dt) within each regime
- Regime lengths follow geometric distribution

Parameters: (σ₀, μ, n_days, samples_per_day)
"""

import numpy as np
import pandas as pd


def simulate_regime_mixture_qvar(
    sigma0,
    mu=0.0,
    n_days=5_000_000,
    samples_per_day=4,
    mean_regime_length_days=None,
    max_window_days=None,
    mean_reversion_rate=0.001,
    seed=None,
):
    """
    Long log-price path with piecewise-constant variance regimes.

    Internal time step: dt_step = 1 / (252 * samples_per_day)
    Total internal steps: n_days * samples_per_day
    Output: one observation per *day* (downsampled).

    Regime model:
        For each regime j:
            tau_j ~ Gamma(3/2, rate = sigma0^2)   # shape=1.5, rate=σ0²
            V_j   = 1 / tau_j                     # variance rate
            ΔL_t  ~ N(mu*dt_step, V_j * dt_step)  # inside the regime

        Regime lengths (in internal steps) are geometric with mean
            mean_regime_length_days * samples_per_day.

    Parameters
    ----------
    sigma0 : float
        Baseline variance scale (appears as σ₀ in the parabola).
    mu : float
        Drift per *year* (kept 0 by default).
    n_days : int
        Number of *trading days* to simulate.
    samples_per_day : int
        Number of internal steps per day (e.g. 4, 24, 78, ...).
    mean_regime_length_days : float or None
        Expected regime length in days. If None, a heuristic is used
        (see `max_window_days` below).
    max_window_days : int or None
        (Optional) Maximum window size in days you care about.
        Used only for the heuristic of `mean_regime_length_days`.
    mean_reversion_rate : float
        Mean reversion rate per year (default: 0.001, giving half-life ≈ 693 days).
        Set to 0.0 to disable mean reversion. We apply mean reversion to prevent
        overflow errors when exponentiating long log-price paths.
    seed : int or None

    Returns
    -------
    prices_daily : ndarray, shape (n_days+1,)
        Price path sampled once per day.
    log_prices_daily : ndarray, shape (n_days+1,)
        Log-price path sampled once per day.
    V_daily : ndarray, shape (n_days,)
        Average variance rate per day (mean of internal V over that day).
    """

    rng = np.random.default_rng(seed)

    # --- internal grid ---
    dt_step = 1.0 / (252.0 * samples_per_day)          # year fraction per internal step
    n_steps = n_days * samples_per_day

    # heuristic for mean regime length if not provided
    if mean_regime_length_days is None:
        if max_window_days is not None:
            # typical heuristic: regimes ≈ 5–10× larger than your max window
            mean_regime_length_days = 10.0 * max_window_days
        else:
            # fallback: something large-ish
            mean_regime_length_days = 2000.0

    mean_regime_length_steps = mean_regime_length_days * samples_per_day
    p_switch = 1.0 / mean_regime_length_steps  # geometric hazard

    # paths on internal grid
    L = np.zeros(n_steps + 1)
    V_path = np.zeros(n_steps)

    # Gamma parameters for precision tau
    alpha = 3/2                      # shape
    beta = sigma0**2                 # rate
    # tau ~ Gamma(alpha, rate=beta) => in numpy: scale=1/beta
    tau = rng.gamma(shape=alpha, scale=1.0 / beta)
    V = 1.0 / tau

    # drift per internal step
    mu_step = mu * dt_step
    
    # We apply mean reversion to prevent overflow errors when exponentiating long log-price paths
    # Mean reversion rate per internal step
    theta_step = mean_reversion_rate * dt_step

    remaining = n_steps
    t = 0
    while remaining > 0:
        # sample regime length in internal steps
        L_reg = rng.geometric(p_switch)
        L_reg = min(L_reg, remaining)

        # simulate this regime
        eps = rng.standard_normal(L_reg)
        
        # Apply mean reversion step-by-step (can't use cumsum due to L dependency)
        for i in range(L_reg):
            # Mean reversion term: -theta * L[t] * dt pulls log-price toward zero
            dL_i = mu_step - theta_step * L[t] + np.sqrt(V * dt_step) * eps[i]
            L[t+1] = L[t] + dL_i
            V_path[t] = V
            t += 1

        remaining -= L_reg

        if remaining > 0:
            # new regime: resample tau, hence V
            tau = rng.gamma(shape=alpha, scale=1.0 / beta)
            V = 1.0 / tau

    # --- downsample to one value per *day* ---
    step = samples_per_day
    L_daily = L[::step]                    # length n_days+1
    prices_daily = np.exp(L_daily)

    # daily average variance over internal steps
    if samples_per_day > 1:
        n_days_eff = len(L_daily) - 1      # should be n_days
        V_daily = V_path[:n_days_eff*step].reshape(n_days_eff, step).mean(axis=1)
    else:
        V_daily = V_path

    return prices_daily, L_daily, V_daily


def simulate_price_path(sigma_f, sigma_n, mu=0.0, S0=100.0, dt=1/252, n_steps=50000, seed=None):
    """
    Simulate a price path using the two-factor Gaussian diffusion model.
    
    Parameters:
    -----------
    sigma_f : float
        Volatility of the fundamental Brownian motion
    sigma_n : float
        Volatility of the noise Brownian motion
    mu : float
        Drift parameter (default: 0.0)
    S0 : float
        Initial price (default: 100.0)
    dt : float
        Time step (default: 1/252 for daily)
    n_steps : int
        Number of time steps to simulate
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    prices : ndarray
        Array of prices S_n = exp(L_n)
    log_prices : ndarray
        Array of log-prices L_n
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize
    X = np.zeros(n_steps + 1)  # Fundamental component
    N = np.zeros(n_steps + 1)  # Noise component
    L = np.zeros(n_steps + 1)  # Log-price
    L[0] = np.log(S0)
    
    # Generate independent standard normals
    xi_f = np.random.randn(n_steps)  # Fundamental noise
    xi_n = np.random.randn(n_steps)  # Noise component
    
    # Simulate using discrete-time equations
    sqrt_dt = np.sqrt(dt)
    for n in range(n_steps):
        X[n+1] = X[n] + sigma_f * sqrt_dt * xi_f[n]
        N[n+1] = N[n] + sigma_n * sqrt_dt * xi_n[n]
        L[n+1] = L[n] + mu * dt + sigma_f * sqrt_dt * xi_f[n] + sigma_n * sqrt_dt * xi_n[n]
    
    # Convert to prices
    prices = np.exp(L)
    
    return prices, L


def generate_price_csv(sigma0, mu=0.0, n_days=5_000_000, samples_per_day=4,
                       max_window_days=130, output_file='variance_timeseries.csv',
                       mean_reversion_rate=0.001, seed=None):
    """
    Generate a CSV file with price data for the Q-Variance challenge using the regime mixture model.
    
    Parameters:
    -----------
    sigma0 : float
        Baseline variance scale (appears as σ₀ in the parabola)
    mu : float
        Drift per year (default: 0.0)
    n_days : int
        Number of trading days to simulate (default: 5_000_000)
    samples_per_day : int
        Number of internal steps per day (default: 4)
    max_window_days : int
        Maximum window size in days for regime length heuristic (default: 130)
    output_file : str
        Output CSV filename
    mean_reversion_rate : float
        Mean reversion rate per year (default: 0.001, giving half-life ≈ 693 days).
        Set to 0.0 to disable mean reversion. We apply mean reversion to prevent
        overflow errors when exponentiating long log-price paths.
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    df : DataFrame
        DataFrame with Price column
    """
    prices, _, _ = simulate_regime_mixture_qvar(
        sigma0=sigma0,
        mu=mu,
        n_days=n_days,
        samples_per_day=samples_per_day,
        mean_regime_length_days=None,
        max_window_days=max_window_days,
        mean_reversion_rate=mean_reversion_rate,
        seed=seed
    )
    
    # Create DataFrame with Price column
    df = pd.DataFrame({'Price': prices})
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Generated {len(df)} price points")
    print(f"Saved to {output_file}")
    
    return df

