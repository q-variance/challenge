"""
Quantized Precision Process for Q-Variance Challenge.

A non-classical stochastic volatility model that quantizes the CIR
precision process onto a discrete lattice, resolving the alpha=3/2
infinite-variance instability that affects continuous-time models.

The precision Z evolves on a discrete lattice via exact CIR transitions
(non-central chi-squared), quantized to the nearest lattice point at
each step. Variance V = sigma0^2 / Z is capped at V_max.

Why alpha = 3/2:
    The q-variance theory predicts E[sigma^2|z] = s0^2 + z^2/2.
    For a CIR precision process, alpha = 3/2 corresponds to k(3/2) = 1/2
    in the cumulant generating function, giving the exact z^2/2 quadratic
    coefficient. Continuous models with alpha = 3/2 have infinite Var(V)
    (InvGamma(3/2) has no finite second moment). Quantizing onto a
    discrete lattice resolves this: Var(V) is finite, enabling stable
    Monte Carlo convergence.

Free parameters (3):
    sigma0 : base volatility (annualized)
    kappa  : mean-reversion speed
    rho    : leverage correlation

Derived constants:
    alpha = 3/2  (from q-variance theory)
    N_max = 200  (structural, insensitive above ~100)

References:
    - Cox, Ingersoll, Ross (1985) "A Theory of the Term Structure"
    - Orrell D (2024) "A Quantum Oscillator Model of Stock Markets", QEF Vol 1 No 1
    - Wilmott P, Orrell D (2025) "Q-Variance", WILMOTT Magazine Vol 138
    - Glasserman (2003) "Monte Carlo Methods in Financial Engineering", Ch. 3.4
"""

import numpy as np
import pandas as pd


class QuantumPrecisionProcess:
    """Quantized CIR precision process for q-variance.

    The precision Z evolves on a discrete lattice via quantized CIR
    transitions. At each step, the exact CIR transition is computed
    (non-central chi-squared) and the result is rounded to the nearest
    lattice point.

    The quantization is essential: it makes Var(V) finite at alpha = 3/2,
    where the continuous InvGamma(3/2) distribution has infinite variance.

    Parameters
    ----------
    sigma0 : float
        Base volatility (annualized). Controls the variance scale.
    kappa : float
        Mean-reversion speed for precision.
    rho : float
        Leverage correlation between precision innovations and returns.
    alpha : float
        Shape parameter. Default 3/2 from q-variance theory.
    N_max : int
        Number of lattice levels (structural). Default 200.
    V_max : float
        Maximum allowed variance (caps extreme values). Default 20.0.
    """

    DEFAULT_ALPHA = 1.5

    def __init__(self, sigma0=0.2691, kappa=1.55, rho=0.41, alpha=None,
                 N_max=200, V_max=20.0):
        self.sigma0 = sigma0
        self.kappa = kappa
        self.rho = rho
        self.N_max = N_max
        self.V_max = V_max
        self.alpha = alpha if alpha is not None else self.DEFAULT_ALPHA

        if self.alpha < 1.0:
            raise ValueError(
                f"alpha={self.alpha:.3f} < 1.0 violates Feller condition."
            )

        self._build_lattice()
        self._precompute_cir(1.0 / 252.0)

    def _build_lattice(self):
        """Build precision lattice covering bulk of Gamma(alpha, 1)."""
        alpha = self.alpha
        N = self.N_max

        Z_max = alpha + 6.0 * np.sqrt(alpha)
        self.dz = Z_max / (N + 0.5)

        ns = np.arange(N + 1)
        self.Z_levels = (ns + 0.5) * self.dz
        self.V_levels = np.minimum(self.sigma0**2 / self.Z_levels, self.V_max)

        log_w = (alpha - 1) * np.log(self.Z_levels) - self.Z_levels
        log_w -= log_w.max()
        self.weights = np.exp(log_w)
        self.weights /= self.weights.sum()

    def _precompute_cir(self, dt):
        """Precompute CIR exact transition parameters."""
        kappa = self.kappa
        alpha = self.alpha
        e_kdt = np.exp(-kappa * dt)
        one_minus_e = 1.0 - e_kdt

        self.cir_c = one_minus_e / 2.0
        self.cir_df = 2.0 * alpha
        self.cir_lam_coeff = 2.0 * e_kdt / one_minus_e
        self.cir_e_kdt = e_kdt

    def _z_to_state(self, Z_val):
        """Map continuous Z to nearest lattice state."""
        n = int(Z_val / self.dz - 0.5 + 0.5)
        return max(0, min(n, self.N_max))

    def simulate(self, n_days, n_tickers=1, dt=1/252, seed=None, burn_in=1000):
        """Simulate price paths with quantized CIR precision.

        Parameters
        ----------
        n_days : int
            Number of trading days per ticker.
        n_tickers : int
            Number of independent tickers.
        dt : float
            Time step in years.
        seed : int or None
            Random seed.
        burn_in : int
            Burn-in steps discarded from output.

        Returns
        -------
        pd.DataFrame
            Columns: Date, Ticker, Price
        """
        rng = np.random.default_rng(seed)
        total_steps = burn_in + n_days

        alpha = self.alpha
        rho = self.rho
        rho_comp = np.sqrt(1.0 - rho * rho)
        Z_levels = self.Z_levels
        V_levels = self.V_levels
        dz = self.dz

        c = self.cir_c
        df = self.cir_df
        lam_coeff = self.cir_lam_coeff
        e_kdt = self.cir_e_kdt
        Z_floor = 1e-8

        all_frames = []

        for ticker_idx in range(n_tickers):
            eps_perp = rng.standard_normal(total_steps)

            Z = rng.gamma(alpha, 1.0)
            state = self._z_to_state(Z)
            Z = Z_levels[state]

            log_prices = np.empty(n_days)
            log_price = 0.0

            for step in range(total_steps):
                Z_pos = max(Z, Z_floor)
                V = V_levels[state]
                daily_vol = np.sqrt(V * dt)

                lam = lam_coeff * Z_pos
                Z_raw = rng.noncentral_chisquare(df, lam)
                Z_new_cont = c * Z_raw

                new_state = self._z_to_state(Z_new_cont)
                Z_new = Z_levels[new_state]

                E_Z_new = alpha * (1.0 - e_kdt) + e_kdt * Z_pos
                var_Z_new = (1.0 - e_kdt) * (
                    alpha * (1.0 - e_kdt) + 2.0 * e_kdt * Z_pos
                )
                std_Z_new = np.sqrt(max(var_Z_new, 1e-16))
                eps_Z = (Z_new_cont - E_Z_new) / std_Z_new

                ret = daily_vol * (rho * eps_Z + rho_comp * eps_perp[step])
                log_price += ret

                if step >= burn_in:
                    log_prices[step - burn_in] = log_price

                Z = Z_new
                state = new_state

            prices = np.exp(log_prices)
            dates = np.arange(n_days)
            df_out = pd.DataFrame(
                {"Date": dates, "Ticker": f"Q{ticker_idx}", "Price": prices}
            )
            all_frames.append(df_out)

        return pd.concat(all_frames, ignore_index=True)

    def state_statistics(self):
        """Return summary statistics of the quantized state distribution."""
        V = self.V_levels
        w = self.weights
        Z = self.Z_levels

        E_V = np.sum(w * V)
        E_V2 = np.sum(w * V**2)
        Var_V = E_V2 - E_V**2
        CV2 = Var_V / E_V**2 if E_V > 0 else 0
        E_Z = np.sum(w * Z)
        mode_idx = np.argmax(w)

        return {
            'alpha': self.alpha,
            'dz': self.dz,
            'N_states': self.N_max + 1,
            'E[V]': E_V,
            'E[Z]': E_Z,
            'Var(V)': Var_V,
            'CV^2': CV2,
            'V_min': V[-1],
            'V_max': V[0],
            'Z_range': (Z[0], Z[-1]),
            'mode_state': mode_idx,
            'mode_Z': Z[mode_idx],
            'mode_V': V[mode_idx],
        }

    def __repr__(self):
        return (
            f"QuantumPrecisionProcess(sigma0={self.sigma0:.4f}, "
            f"kappa={self.kappa:.4f}, rho={self.rho:.4f}, "
            f"alpha={self.alpha:.4f}, N_max={self.N_max})"
        )
