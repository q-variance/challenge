# Stochastic Volatility Model Submission

## Model Overview

The **Stochastic Volatility Model** describes the evolution of a security price $S$ and its instantaneous variance $V = \sigma^{2}$, where $\sigma$ is the volatility. The continuous-time stochastic processes are defined as:
- $dS = \mu S dt + \sigma S dw$
- $dV = \phi V dt + \xi V dz$

In this specific implementation, we assume the **volatility of volatility** is zero ($\xi = 0$). The drift of variance, $\phi$, is driven by the absolute log-return of the asset: $\phi = b \times |\Delta \ln(S)|$, where $b$ is a free scaling parameter.

## Parameters

There are 3 free parameters in the model:
- Initial Annual Volatility: $\sigma = 0.15$
- Annual Expected Return: $\mu = 0.02$
- Volatility Scale Factor: $b = 0.0225$

## Simulation
The simulation consists of 500 samples, each spanning 10,000 days. Starting from day 3, the discrete-time updates are calculated as follows:

- Variance Update:
$V_i = V_{i-1} \exp\left( b \cdot |\ln(S_{i-1} / S_{i-2})| \right)$
- Price Update:
$S_i = S_{i-1} \exp\left( \mu - \sigma_i^{2}/2 + \sigma_i \epsilon \right)$

Note:
* $\sigma_i = \sqrt{V_i}$
* $\epsilon\sim \mathcal{N}(0,1)$ 
* Data generation logic can be found in the [price_generator.ipynb](price_generator.ipynb).

## Results
The model achieves a high $R^{2}$ with the simulated paths.
* **Global R²**: **0.998**

<img src="Figure_1.png" width="700">
<img src="Figure_3.png" width="700">