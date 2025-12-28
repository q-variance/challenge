# GARCH(1,1) Volatility Model Submission

## Model Overview
The **GARCH(1,1) Volatility Model** (Generalized Autoregressive Conditional Heteroskedasticity) describes the evolution of a security price $S$ where the variance $V$ is a stochastic process exhibiting **Mean Reversion** and **Volatility Clustering**. 

This implementation maps the discrete-time GARCH process to the theoretical **Q-Variance** relationship. By simulating millions of trading days, we demonstrate that the stochastic variance updates converge to a deterministic parabolic fit: 
$$V(z) = \sigma_0^2 + \frac{(z - z_{off})^2}{2}$$



## Parameters & Mapping Logic
The model utilizes four parameters to control the GARCH dynamics and recover the theoretical geometry of the Q-variance parabola:

| Parameter | Value | Influence on Q-Variance Geometry |
| :--- | :--- | :--- |
| **Target Vol ($\sigma$)** | **0.425** | **Minimal Volatility ($\sigma_0$):** Sets the vertical baseline (the "floor" of the parabola). |
| **Annual Return ($\mu$)** | **0.06** | **Z-Shift ($z_{off}$):** Controls the horizontal asymmetry (displacement from zero). |
| **Persistence ($\beta$)** | **0.8** | **Curvature/Steepness:** Determines the rate of decay; higher $\beta$ maintains "memory" of past volatility. |
| **Shock Weight ($\alpha$)** | **0.19** | **Sensitivity:** Controls the reaction to market shocks. Fixed such that $\alpha + \beta = 0.99$. |

> **Note:** The "Mean Reversion" strength is implicitly defined as $\gamma = 1 - (\alpha + \beta) = 0.01$.

## Simulation Methodology
The simulation generates a synthetic price history using independent paths of **2,500 trading days** each. To eliminate "local path luck" and ensure statistical smoothing, we utilized **2,000 samples** to create a total dataset of **5,000,000 trading days**.

### Discrete-Time Updates

- **Variance Update (GARCH Logic):**
The variance at step $i$ is driven by the weighted average of the long-run variance (Variance Targeting), the previous variance, and the most recent market percentage-return shock ($\epsilon_{i-1}$):

$$V_i = \omega + \alpha \cdot \epsilon_{i-1}^2 + \beta \cdot V_{i-1}$$

Where:
- $\epsilon_{i-1} = \frac{S_{i-1} - S_{i-2}}{S_{i-2}}$
- $\omega = \frac{\sigma^{2}}{252} \cdot 0.01$ (Targeting a long-run annual volatility $\sigma$)
- $\alpha = 0.99 - \beta$



- **Price Update:**
The price follows a Geometric Brownian Motion (GBM) step, adjusted for the current stochastic variance $V_i$:

$$S_i = S_{i-1} \exp\left( \frac{\mu}{252} - \frac{V_i}{2} + \sqrt{V_i} Z \right)$$

where $Z \sim \mathcal{N}(0,1)$.

---

## Results & Convergence Analysis

The model was optimized using the 5M-day horizon to achieve a high-fidelity fit against the theoretical target parameters ($\sigma_0 = 0.2586$ and $z_{off} = 0.0214$).

### Optimized Model Performance
| Total Days | $\sigma_0$ (Target) | $z_{off}$ (Target) | $R^2$ | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **5,000,000** | 0.2586 | 0.0214 | **0.997** | High-fidelity structural convergence. |
| **100,000** | 0.2586 | 0.0214 | **0.966** | Lower $R^2$ due to idiosyncratic noise. |

### Comparative Visualizations
The transition from 100k days to 5M days shows the "clearing" of statistical noise. In the large sample, the stochastic variance updates converge almost perfectly to the theoretical parabola.



#### **100,000 Day Sample (Noise-Dominant)**
<div style="display: flex; justify-content: space-around;">
  <img src="Figure_1_100k.png" width="32%">
  <img src="Figure_3_100k.png" width="32%">
  <img src="Figure_4_100k.png" width="32%">
</div>

#### **5,000,000 Day Sample (Structural Convergence)**
<div style="display: flex; justify-content: space-around;">
  <img src="Figure_1_5M.png" width="32%">
  <img src="Figure_3_5M.png" width="32%">
  <img src="Figure_4_5M.png" width="32%">
</div>

---

### Statistical Stability Analysis
By plotting $R^2$ as a function of total simulated days, we identified a clear threshold for statistical validity:

- **Convergence Point:** The model consistently reaches $R^2 > 0.995$ after approximately **1,130,000 days**.
- **The Law of Large Numbers:** Beyond 2 million days, the fit quality stabilizes at an asymptote, confirming that the optimized GARCH parameters accurately represent the underlying Q-variance structure.

![Convergence Analysis](convergence_analysis.png)
Figure: $R^2$ score vs. Total Days. The 0.995 threshold is maintained after the 1M-day mark.

---
## Acknowledgments & Version History

> **Version 2.0 Update**
> An earlier version of this model utilized an incorrect definition of the intercept term ($\omega$), which led to a scaling mismatch in the long-run variance. Special thanks to **@Orrell** for pointing out this discrepancy.

In this current version, I have:
* **Fixed the $\omega$ definition:** Implemented proper **Variance Targeting** logic, where $\omega = V_{target} \cdot \gamma$.
* **Standardized Nomenclature:** Renamed parameters to follow standard GARCH(1,1) conventions ($\alpha$ for the shock component and $\beta$ for the persistence component).
* **Re-optimized Parameters:** Recalibrated the parameter set specifically for the 5M-day horizon to ensure maximum fit.

Despite these structural corrections, the results remain consistent with previous findings, further confirming the original hypothesis of **Q-variance emergence** in GARCH-governed systems.

---

## Project Structure
- [price_generator2.ipynb](price_generator2.ipynb): Vectorized simulation and optimization logic.
- [simulated_prices.csv](simulated_prices.csv): 100k days of raw price data for verification.
- `dataset_part1.parquet`, `dataset_part2.parquet`, `dataset_part3.parquet`: 5M days of analyzed window data.