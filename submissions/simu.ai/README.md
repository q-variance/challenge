# simu.ai

**Team**: simu.ai

## Results

<img src="Figure_1.png" width="700">

## Model Description

This submission implements a **regime-switching variance model** where the log-price follows a piecewise-constant variance process with regime changes.

**This is a two-parameter model**  using:
- **σ₀** = 0.25 (baseline variance scale)
- **μ** = 0.02 (drift per year, corresponds to z₀ in the q-variance formula)

### Model Structure

The model uses a regime mixture approach where:
- **Regime-switching variance**: V_j = 1/τ_j where τ_j ~ Gamma(3/2, rate=σ₀²)
- **Log-price increments**: ΔL_t ~ N(μ·dt, V_j·dt) within each regime
- **Regime lengths**: Follow geometric distribution with mean ≈ 10 × max_window_days

The model produces the q-variance relationship: σ²(z) = σ₀² + (z - z₀)²/2

### Model Parameters

- **σ₀** = 0.25 (baseline variance scale)
- **μ** = 0.02 (drift per year, corresponds to z₀)

### Simulation Settings

- **n_days** = 5,000,000 (simulation length)
- **samples_per_day** = 4 (internal simulation granularity)
- **max_window_days** = 130 (maximum window size for regime length heuristic)

### Implementation

The model is implemented in `model_simulation.py` and can be regenerated using `generate_submission.py`. The simulation generates a long time series of daily prices, which is then processed through the challenge's data loader to produce the `dataset.parquet` file.

### Time-Invariance

The model demonstrates time-invariance across different period lengths T, as shown in [Figure_5](Figure_5.png), where the distribution of scaled log-returns z remains consistent across different time horizons.

### Dependencies

- numpy
- pandas
- scipy (for data processing)

## Contact

(Optional - add contact information if desired)
# test
