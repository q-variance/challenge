# EquityQuant.dev Model – Event-Driven Price Dynamics

**Team name:** equityquant.dev  
**Contact:** Yi-Lung (Dragon) Tsai — ytsai@equityquant.dev

---

## Model Summary

This submission implements a **classical, continuous-time, event-driven price process**.  
The model is fully specified by its Python implementation and **generates the q-variance relationship endogenously** from the underlying price dynamics, without imposing a quadratic form by construction.

Price dynamics are driven by **stochastic shock activity** rather than by a diffusive volatility process. Volatility is **not modelled directly**; instead, it emerges naturally from the realised aggregation of random shocks.

---

## Simulation Overview

Price paths are generated via the following mechanism:

1. Simulate a latent precision process  
2. Map precision to stochastic shock intensity  
3. Aggregate a random number of Gaussian shocks per day  

This construction ensures that:

- Conditional variance increases with realised price movement  
- The q-variance relationship emerges naturally across time horizons  

---

## Latent Precision Process

A positive latent process $\( \tau_t \)$ governs market activity.  
It is simulated using a **stationary mean-reverting positive diffusion** (CIR form, used purely for numerical convenience):

$$
d\tau_t = \kappa(\theta - \tau_t)\,dt + \eta \sqrt{\tau_t}\, dW_t
$$


### Interpretation

- High $\tau$ → calm market regimes with low activity  
- Low $\tau$ → turbulent regimes with elevated activity  

Mean reversion ensures regime persistence while preventing degeneracy.  
This latent process does **not** represent volatility.


---

## Shock Intensity and Returns

### Shock Intensity

Daily shock intensity is defined as an inverse function of precision:

$$
\lambda_t = \frac{c}{\tau_t}
$$

The number of shocks per day is drawn as:

$$
N_t \sim \text{Poisson}(\lambda_t)
$$

---

### Return Generation

Conditional on $N_t$, daily log-returns are generated as:

$$
r_t \mid N_t \sim \mathcal{N}\left(0,\ s_{\text{unit}}^2 \cdot N_t\right)
$$

This implies:

- Variance is proportional to realised shock activity  
- Returns are Gaussian *conditional* on $N_t$  
- Returns are **heavy-tailed unconditionally**, consistent with subordinated stochastic processes  

---

## Volatility Calibration

The per-shock variance $s_{\text{unit}}^2$ is calibrated such that the unconditional long-run daily variance satisfies:

$$
\mathbb{E}[r_t^2] = \mathbb{E}[N_t] \cdot s_{\text{unit}}^2 = \frac{\sigma_0^2}{252}
$$

This ensures that:

- $\sigma_0$ represents the **minimum volatility level**  
- Corresponds to calm market regimes

No additional free parameter is introduced at this stage;  
the per-shock variance is fully determined by $\( \sigma_0 \)$ and the stationary mean intensity.

---

## Price Process

Log-prices evolve via cumulative daily returns:

$$
\log P_t = \log P_{t-1} + r_t
$$

$$
P_t = \exp(\log P_t)
$$

This guarantees:

- Strictly positive prices  
- Correct aggregation of returns across time  

---

## Parameters

### Free Parameters

The model uses **three effective free parameters**:

| Parameter | Value | Description |
|---------|------|------------|
| $\sigma_0$ | 0.25 | Long-run annual volatility scale |
| $\kappa$ | 0.02 | Mean-reversion speed of precision |
| $c$ | 10.0 | Shock intensity scaling |

These parameters control the scale and persistence of shock activity and were **not tuned** to fit the q-variance parabola.

---

### Fixed Parameters

All remaining parameters are fixed *a priori* for numerical stability and scale normalisation and were **not tuned** to achieve q-variance:

| Parameter | Value | Description |
|---------|------|--------|
| `a_shape` | 1.5 | Diffusion discretisation constant |
| `lam_cap` | 500.0 | Poisson intensity cap (numerical safeguard) |
| `dt` | 1 / 252 | Trading-day discretisation |
| `seed` | 3 | Reproducibility only |
| `s0` | 100.0 | Initial price |
| `n_days` | 120,000 | Simulation length |

The Poisson cap is set sufficiently high that it is **rarely binding** and does not affect the fitted q-variance curve.

---

## Results

### Q-Variance Relationship

<p align="center">
  <img src="figures/result1.png" width="650">
</p>

The simulated price paths reproduce the q-variance relationship with a global
goodness-of-fit of **R² ≈ 0.996**, exceeding the challenge threshold.


