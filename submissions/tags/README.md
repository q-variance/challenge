# tags – IG(3/2, σ₀²) volatility model

## Model summary

This model simulates a single asset price as a GBM with **daily stochastic variance** drawn from an Inverse-Gamma distribution with shape 3/2:

- Instantaneous variance per year on day *t*:
  \[
  V_t \sim \mathrm{InvGamma}(\alpha = 3/2,\ \beta = \sigma_0^2)
  \]
- Daily time step:
  \[
  dt = \frac{1}{252}
  \]
- Conditional on \(V_t\), the daily log-return is
  \[
  \Delta \log S_t
  = (\mu - \tfrac{1}{2} V_t)\,dt + \sqrt{V_t\,dt}\,\varepsilon_t,
  \quad \varepsilon_t \sim \mathcal{N}(0,1).
  \]
- The price path is obtained as \(S_{t+1} = S_t \exp(\Delta \log S_t)\).

This is kinda of classical Normal–Inverse-Gamma variance-mixture structure which yields the q-variance relation for a single observation
\[
\mathbb{E}[V \mid z] = \sigma_0^2 + \tfrac{1}{2} z^2
\]
when \(\alpha = 3/2\) and \(\beta = \sigma_0^2\), and empirically reproduces the canonical q-variance curve with \(R^2 \approx 0.9988\) on the full 100k-day simulation.

- Shape of IG: \(\alpha = 3/2\) (fixed by theory, not tuned).
- Scale of IG: \(\sigma_0 = 0.259\) (q-variance intercept).
- Time step: \(dt = 1/252\).

## Drift calibration and log-drift cancellation

In the discrete-time SDE above,
\[
\mathbb{E}[\Delta \log S_t]
= (\mu - \tfrac{1}{2}\mathbb{E}[V_t])\,dt.
\]

For \(V_t \sim \mathrm{InvGamma}(3/2,\sigma_0^2)\) in the shape–scale parametrisation,
\[
\mathbb{E}[V_t] = \frac{\sigma_0^2}{\alpha - 1}
= 2\sigma_0^2.
\]

We therefore choose
\[
\mu = \sigma_0^2
\]
so that
\[
\mathbb{E}[\Delta \log S_t] = 0,
\]
i.e. the process has **zero expected log-drift** over long horizons.  

**NOTE** In this version of the model, I explicitly account for the 1/252 time step and avoids the systematic long-run decay that would result from taking \(\mu = 0\) with the same volatility structure. The q-variance diagnostics use drift-adjusted returns, so this choice of \(\mu\) affects the **visual behaviour of the price path** but not the underlying q-variance structure.

## Files in this folder

- **`tags_IG32_100Kdays_sim.csv`**  
  Raw daily price path \(S_t\) from the tags IG(3/2, \(\sigma_0^2\)) model, **100,000 trading days**, single asset, one column `Price` (start \(S_0 = 100\)). This is the “longer simulation” requested for convergence / illustration.

- **`tags_IG32_10Kdays_sim.csv`**  
  The **first 10,000 days** of `tags_IG32_100Kdays_sim.csv`, one column `Price`. This file is used as input to `data_loader_csv.py` for the 10k-window diagnostics.

- **`dataset.parquet`**  
  Window-level q-variance dataset generated from the **100,000-day** simulation (`tags_IG32_100Kdays_sim.csv`) using `code/data_loader_csv.py`. Same column structure as the main competition dataset (T, z, sigma, etc.). This is the dataset that yields the full-sample q-variance fit with \(R^2 \approx 0.9988\).

- **`dataset_10K.parquet`**  
  Same construction as `dataset.parquet`, but using **only the first 10,000 days** of the simulation (`tags_IG32_10Kdays_sim.csv`). This is the input for the q(T) vs T diagnostic.

- **`tags_qT_first10k.png`**  
  Plot of the **best-fit quadratic coefficient** \(q(T)\) versus horizon \(T\), where for each T we fit
  \[
  \sigma^2(z;T) = \sigma_0^2 + q(T)\,(z - z_0)^2
  \]
  with \(\sigma_0 = 0.259\) and \(\ z_0 = 0.021\) fixed, using only windows from `dataset_10K.parquet` (first 10k days). The dashed line at 0.5 shows the theoretical q-variance coefficient.

## Interpretation of the 10k-day q(T) plot

The `tags_qT_first10k.png` figure shows that, using only the first 10,000 days of a single simulated path, the estimated quadratic coefficient \(q(T)\) lies in a relatively tight band (roughly 0.28–0.38) and is **approximately stable in T**, i.e. there is no strong systematic horizon dependence. This is the desired scale-invariance feature of the q-variance structure.

The coefficients are systematically below the theoretical value 0.5, reflecting finite-sample effects and the fact that \(\sigma_0\) and \(z_0\) are held fixed while only \(q(T)\) is fitted on a single realisation. When the full 100k-day simulation is used (`dataset.parquet`), the global q-variance fit converges back to the canonical curve with \(R^2 \approx 0.9988\), consistent with the theoretical coefficient of 0.5.

