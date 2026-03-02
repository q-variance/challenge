# Submission: Quantized Precision Process

**Author:** Paul Symonds

## Model Description

A **quantized CIR precision process** that resolves the alpha = 3/2 infinite-variance instability identified in the competition summary.

The precision Z = 1/V evolves via exact CIR transitions on a discrete lattice. At each daily step, the continuous CIR transition (non-central chi-squared) is computed and quantized to the nearest lattice point. Variance is then V = sigma_0^2 / Z, capped at V_max.

**Why it works:** The competition proved that any continuous-time model satisfying q-variance requires alpha = 3/2, which makes Var(V) infinite. Quantizing precision onto a discrete lattice makes Var(V) finite while preserving the alpha = 3/2 dynamics. The discretization is not a numerical convenience -- it is essential for convergence.

## Parameters (3 free)

| Parameter | Value  | Role |
|-----------|--------|------|
| sigma_0   | 0.2691 | Base volatility (annualized) |
| kappa     | 1.55   | Mean-reversion speed |
| rho       | 0.41   | Leverage correlation |

**Fixed by theory:**
- alpha = 3/2 (from q-variance: k(3/2) = 1/2 gives the z^2/2 coefficient)
- N_max = 200 (structural; insensitive above ~100)

## Results

- **R^2 = 0.998** (seed 42, 5M days) against the target parabola
- **8/8 seeds pass** R^2 >= 0.995 at 5M days (mean 0.997, min 0.996)
- **Convergent:** R^2 = 0.94 (100K) -> 0.98 (500K) -> 0.997 (1M) -> 0.998 (5M)
- **Time-invariant:** consistent across all horizons T = 5 to 130
- **500-segment test:** all 500 segments processable

## Mathematical Framework

Precision lattice:
```
Z_n = (n + 0.5) * dZ,    n = 0, 1, ..., N_max
V_n = min(sigma_0^2 / Z_n, V_max)
```

CIR dynamics (quantized):
```
dZ = kappa * (alpha - Z) * dt + sqrt(2 * kappa * Z) * dW
```

Returns with leverage:
```
r_t = sqrt(V_t / 252) * (rho * eps_Z + sqrt(1 - rho^2) * eps_perp)
```

## How to Reproduce

```bash
# Generate full submission (5M days, ~10s simulation + ~30s windowing)
python generate_submission.py

# Quick test (500K days)
python simulate.py --days 500000 --seed 42
```

Requirements: Python 3.10+, numpy, pandas, scipy (for scoring only)

## Files

| File | Description |
|------|-------------|
| `model.py` | Standalone QuantumPrecisionProcess class |
| `simulate.py` | CLI simulation driver |
| `generate_submission.py` | End-to-end pipeline (simulate + window + score) |
| `dataset.parquet` | 5M-day windowed dataset (3.85M windows) |
| `prices_100k.csv` | Sample 100K daily prices |
| `quantized_precision_paper.md` | Full technical paper |

## Contact

Paul Symonds
