# EquityQuant.dev Model 

**Team name:** equityquant.dev  
**Model Description:** please see below

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
| $\kappa$ | 0.5 | Mean-reversion speed of precision |
| $c$ | 10.0 | Shock intensity scaling |

Note: In the latest code version, 
𝜅 is set in a **fast-mixing regime** to improve stability and reduce sensitivity to simulation length.

---

### Fixed Parameters

All remaining parameters are fixed *a priori* for numerical stability and scale normalisation and were **not tuned** to achieve q-variance:

| Parameter | Value | Description |
|---------|------|--------|
| `a_shape` | 1.5 | Diffusion discretisation constant |
| `lam_cap` | 500.0 | Poisson intensity cap (numerical safeguard) |
| `dt` | 1 / 252 | Trading-day discretisation |
| `burn_in` | 2000  | burn-in removed from output |
| `seed` | 6 | Reproducibility only |
| `s0` | 100.0 | Initial price |

The Poisson cap is set sufficiently high that it is **rarely binding** and does not affect the fitted q-variance curve.

---

## Results

ex: n_days = 100K 
<p align="center">
  <img src="result1.png" width="650">
</p>

| Simulation Length (n_days) | σ₀ (fitted) | z₀ (fitted) |    R² |
| -------------------------: | ----------: | ----------: | ----: |
|                      5,000 |       0.346 |       0.048 | 0.947 |
|                     10,000 |       0.344 |       0.054 | 0.970 |
|                     15,000 |       0.315 |       0.055 | 0.972 |
|                     60,000 |       0.318 |       0.002 | 0.991 |
|                    100,000 |       0.323 |      −0.012 | 0.997 |
|                    120,000 |       0.318 |      −0.017 | 0.997 |
|                    180,000 |       0.297 |      −0.012 | 0.990 |


## Dataset Notes

Due to GitHub file size limitations, the submission dataset is provided in three parts:
`dataset_part1.parquet`, `dataset_part2.parquet`, and `dataset_part3.parquet`.

This format is fully supported by the official `score_submission.py` script, which
automatically concatenates the split files when `dataset.parquet` is not present.
No data has been modified or filtered; the three files together form the complete
submission dataset.

## Contact 
Thelonious Casey — tcasey@equityquant.dev

