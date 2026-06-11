# Three-parameter noncommutative bath model for q-variance

This submission gives a **three-parameter path generator** for the q-variance challenge. The model is designed to produce a full price path, not only a static fitted curve.

The submitted parameter set is:

| parameter | value |
|---|---:|
| `beta_mult` | 2.028391 |
| `memory` | 110.393674 |
| `eta` | 1.091468 |

The local 5M optimisation gives a pooled official fixed-parabola score above `0.995`:

| official-target metric | value |
|---|---:|
| `fixed_mean` | 0.995813 |
| `fixed_min` | 0.995337 |
| `fixed_max` | 0.996453 |
| `fixed_std` | 0.000400 |

This is the main challenge result. The current run is above `0.995` both in the mean score and in the worst-seed score.

## Main claim

The model reaches the challenge target: pooled official fixed q-variance R² above `0.995`.

In addition, it gives a more T-stable empirical fit than the fixed parabola. The fixed q-variance parabola is T-invariant as a formula, but its empirical goodness-of-fit is not equally stable across individual window lengths. The proposed path model gives a more stable fit to the empirical q-variance surface `Q(z,T)`.

## Model in one paragraph

The process uses a persistent hidden activity bath. The daily innovation is decomposed into an even activity channel and an odd signed-pressure channel. Two order-flow combinations of those channels form a hidden state. That state controls the daily variance multiplier. The generator has exactly three numerical parameters: `beta_mult`, `memory`, and `eta`.

## Definition

Let `eps_t` be an i.i.d. standard normal innovation. Define

```text
E_t = (eps_t^2 - 1) / sqrt(2)
O_t = eps_t
```

The baseline bath driver is

```text
D_t = E_t - tanh(eta) O_t
```

and the persistent bath state is

```text
F0_t = AR1(D_(t-1); memory)
```

The two order-flow channels are

```text
C_t = E_(t-1) O_(t-2) - O_(t-1) E_(t-2)
S_t = E_(t-1) O_(t-2) + O_(t-1) E_(t-2)
```

After removing overlap with the baseline bath, the submitted hidden state is

```text
F_t = std(F0_t + C_t_perp - tanh(eta) S_t_perp)
```

The activity multiplier is

```text
A_t = exp(eta F_t) / mean(exp(eta F_t))
```

and returns are generated as

```text
r_t = sqrt(beta_mult * sigma0^2 * A_t / 252) * eps_t
```

## Market interpretation

The model can be read as a simple market-state model. Prices do not move only because of today’s random shock; they move inside a market environment that remembers recent pressure, activity, and imbalance.

Each daily innovation is split into two parts. One part measures how intense the day is, regardless of direction: a quiet day, an ordinary day, or a high-information day. The other part keeps the sign of the move: whether the pressure is upward or downward. The hidden bath is a persistent memory of these recent signed and unsigned shocks.

This hidden state represents market activity or pressure. When the state is high, the market is more active and daily variance is higher. When the state is low, the market is quieter. The parameter `memory` controls how long this state persists, `eta` controls how strongly it affects volatility, and `beta_mult` sets the overall variance scale.

The noncommutative/order-flow part means that the order of events matters. A large activity shock followed by directional pressure is not treated as identical to directional pressure followed by a large activity shock. This is meant to capture a simple market fact: volatility, liquidity, and directional pressure do not commute in real trading. The same ingredients can have different effects depending on their sequence.

This produces q-variance because the endpoint move over a window and the realised variance inside that window are driven by the same persistent hidden state. A large endpoint move is therefore more likely to have occurred during a period of elevated market activity, which raises the conditional realised variance. The model is not just fitting a static parabola; it generates full price paths whose realised variance and endpoint displacement are coupled through a persistent market state.

In this interpretation, the official q-variance parabola is the pooled statistical signature, while the model is a possible dynamic mechanism behind it.

## Official score

The official target is the pooled fixed q-variance curve. With the parameter set above, the current local 5M result is:

```text
fixed_mean = 0.995813
fixed_min  = 0.995337
```

This is the result intended for the challenge ranking.

## T-invariance and empirical window dependence

David Orrell's fixed q-variance parabola is T-invariant by construction. Empirically, however, its fit varies substantially across individual window lengths. The proposed process gives a more stable per-window empirical fit.

### Central range: |z| < 0.6

| T | model vs empirical | Orrell's parabola vs empirical | model gain |
|---:|---:|---:|---:|
| 5 | 0.978292 | 0.739738 | 0.238555 |
| 10 | 0.981181 | 0.977943 | 0.003238 |
| 20 | 0.981770 | 0.986658 | -0.004889 |
| 40 | 0.970829 | 0.939829 | 0.031000 |
| 80 | 0.973695 | 0.860966 | 0.112730 |
| 130 | 0.977794 | 0.846932 | 0.130862 |

Summary across all T slices:

| metric | model | Orrell's parabola |
|---|---:|---:|
| mean per-T R² | 0.974796 | 0.885335 |
| min per-T R² | 0.966724 | 0.739738 |
| std across T | 0.004780 | 0.066584 |
| range across T | 0.015046 | 0.246921 |
| T-slices won by model | 13/14 | — |

![Empirical per-T fit, |z|<0.6](fig_empirical_perT_r2_z06.png)

Selected q-variance slices:

![q-variance slice T=5](fig_qvariance_T5_z06.png)

![q-variance slice T=20](fig_qvariance_T20_z06.png)

![q-variance slice T=80](fig_qvariance_T80_z06.png)

![q-variance slice T=130](fig_qvariance_T130_z06.png)

### Wider robustness range: |z| < 1.0

| T | model vs empirical | Orrell's parabola vs empirical | model gain |
|---:|---:|---:|---:|
| 5 | 0.986482 | 0.879179 | 0.107303 |
| 10 | 0.985774 | 0.988109 | -0.002335 |
| 20 | 0.985602 | 0.991417 | -0.005815 |
| 40 | 0.977375 | 0.972311 | 0.005064 |
| 80 | 0.972307 | 0.944460 | 0.027847 |
| 130 | 0.947315 | 0.921455 | 0.025861 |

Summary across all T slices:

| metric | model | Orrell's parabola |
|---|---:|---:|
| mean per-T R² | 0.965389 | 0.942759 |
| min per-T R² | 0.933024 | 0.879179 |
| std across T | 0.017000 | 0.033750 |
| range across T | 0.053458 | 0.112238 |
| T-slices won by model | 11/14 | — |

![Empirical per-T fit, |z|<1.0](fig_empirical_perT_r2_z10.png)

The wider range is not the main optimisation target; it is included as a robustness check.

## Local scoring commands

Generate the official price CSV:

```powershell
python strict_three_param_noncomm_antisym_bath_model.py --mode sym_neg_tanh --n 5000000 --seed 1 --beta-mult 2.028391 --memory 110.393674 --eta 1.091468 --out-prefix final_symneg_5M --out-price .\score_run\variance_timeseries.csv --quiet
```

Run the official conversion and scorer from the `score_run` folder:

```powershell
cd .\score_run
python data_loader_csv.py
python score_submission.py
cd ..
```

`data_loader_csv.py` reads `variance_timeseries.csv` and writes `dataset.parquet`. `score_submission.py` scores `dataset.parquet`.

To regenerate a 100K sample price CSV for the submission folder:

```powershell
python strict_three_param_noncomm_antisym_bath_model.py --mode sym_neg_tanh --n 100000 --seed 1 --beta-mult 2.028391 --memory 110.393674 --eta 1.091468 --out-prefix sample_symneg_100k --out-price .\submissions\katastrofa_noncomm\sample_100k_prices.csv --quiet
```

## Files for PR

Expected submission folder:

```text
submissions/katastrofa_noncomm/
    README.md
    dataset.parquet
    sample_100k_prices.csv
    strict_three_param_noncomm_antisym_bath_model.py
    fig_empirical_perT_r2_z06.png
    fig_empirical_perT_r2_z10.png
    fig_qvariance_T5_z06.png
    fig_qvariance_T20_z06.png
    fig_qvariance_T80_z06.png
    fig_qvariance_T130_z06.png
```

## Notes before final PR

- Add the exact output of `score_submission.py`.
- If a 7-seed rerank is used, update the score table.
- Keep the README focused on the official pooled R² above `0.995` first, then the empirical T-stability evidence.
