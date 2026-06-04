# Coherent inverse-chi-square Hermite-energy price generator

This submission contains a three-parameter stochastic price generator for the Q-Variance Challenge. The generator produces a single daily price path and writes it as `variance_timeseries.csv` with one column named `Price`, ready for the public challenge scripts.

The model treats market variance as a persistent activity field. The activity has two components:

1. a persistent inverse-chi-square field, representing slowly changing market activity regimes;
2. a coherent Hermite-energy feedback field, representing the way large shocks leave a trace in subsequent realised variance.

The construction is a daily time-series model. The same generated path is used by the scorer at all horizons.

## Calibrated parameters

The submitted calibration uses three fitted parameters:

```text
beta_mult = 1.719800
memory    = 15.275991
eta       = 0.498969
```

Their roles are:

```text
beta_mult   annual variance scale, expressed relative to the challenge volatility unit
memory      persistence scale of the latent activity fields, in trading days
eta         strength of the Hermite-energy feedback
```

The random seed controls reproducibility of the submitted price path.

## Model definition

Let

\[
\epsilon_t \sim N(0,1)
\]

be the daily innovation. The persistence coefficient is

\[
\rho = \exp(-1/m),
\]

where \(m\) is the `memory` parameter.

### Persistent market activity

Generate a persistent Gaussian field

\[
U_t = \rho U_{t-1} + \sqrt{1-\rho^2}\,\xi_t,
\qquad \xi_t \sim N(0,1).
\]

Map it through a Gaussian copula into a three-dimensional inverse-radius activity variable:

\[
p_t = \Phi(U_t),
\qquad
Q_t = F^{-1}_{\chi^2_3}(p_t),
\]

\[
A^{(0)}_t = \frac{1}{Q_t}.
\]

The activity series is normalized to sample mean one. The three-dimensional inverse-radius field is the fixed activity geometry used by this model.

### Hermite-energy feedback

The even Hermite energy of the daily innovation is

\[
H_2(\epsilon_t)=\epsilon_t^2-1.
\]

The feedback driver is

\[
D_t =
\frac{H_2(\epsilon_t)}{\sqrt{2}}
-
\frac{\eta^2}{\sqrt{2}}\epsilon_t.
\]

The odd part is tied to the same coupling parameter \(\eta\). Its role is to allow a small coherent shift of the conditional-variance curve while keeping the feedback controlled by a single strength parameter.

After standardization, the driver is passed through the same persistence filter:

\[
F_t = \rho F_{t-1}+\sqrt{1-\rho^2}\,\widetilde D_t,
\]

where \(\widetilde D_t\) denotes the standardized driver. The feedback multiplier is

\[
M_t = \exp(\eta F_t),
\]

again normalized to sample mean one.

### Returns and prices

The total activity is

\[
A_t =
\frac{A^{(0)}_t M_t}{\langle A^{(0)}M\rangle}.
\]

The daily log return is

\[
r_t =
\sqrt{
\frac{
\beta_{\rm mult}\sigma_0^2 A_t
}{252}
}
\,\epsilon_t,
\]

with the sample mean removed. The value \(\sigma_0=0.2586\) is used as the volatility unit of the public challenge benchmark; equivalently the free annual variance scale is

\[
\beta = \beta_{\rm mult}\sigma_0^2.
\]

The price path is

\[
P_t =
P_0\exp\left(\sum_{s\le t}r_s\right),
\]

with a centering of the accumulated log-price used for numerical stability.

## Market interpretation

The inverse-chi-square activity field represents uneven market conditions: periods of ordinary trading activity mixed with occasional high-activity regimes. This is a compact way to model persistent variation in liquidity, positioning, leverage pressure, margin constraints and order-flow imbalance.

The Hermite-energy feedback represents the empirical fact that large shocks are not isolated events in a realistic price path. Large innovations tend to occur in market states where realised variance is elevated, and they also affect subsequent activity. The feedback term links daily shock energy to the activity process, producing a stronger relation between endpoint moves and realised variance than a conditionally independent volatility mixture.

## Output diagnostics

The following figures were generated from a 5,000,000-day simulation with the calibrated parameters above and seed 5.

### Pooled q-variance curve

![Pooled q-variance curve](figures/qvariance_pooled.png)

The pooled fixed-parabola score over the public scoring window \(|z| < 0.6\) is:

```text
R2 = 0.996831
```

### Per-horizon q-variance diagnostics

![Per-horizon q-variance diagnostics](figures/qvariance_by_T.png)

Diagnostic \(R^2\) values over \(|z| < 1.0\), using \(T=5,10,20,40,80\):

```text
T = 5    R2 = 0.988047
T = 10   R2 = 0.996026
T = 20   R2 = 0.983928
T = 40   R2 = 0.984238
T = 80   R2 = 0.858349
```

### Scaled-return density by horizon

![Scaled-return density by horizon](figures/z_density_by_T.png)

### Example generated price path

![Example generated price path](figures/price_path_sample.png)

### Sensitivity to the scoring window in z

![Sensitivity to z boundary](figures/zrange_sensitivity.png)

## Files

```text
coherent_ig_energy_submission.py     model and price-series generator
variance_timeseries_100k.csv         short sample for checking the CSV format
model_parameters.json                calibrated parameter values
requirements.txt                     Python dependencies
internal_validation_report.txt       local validation notes
validation_metrics.txt               metrics used for the figures in this README
figures/                             output figures from the 5,000,000-day diagnostic run
```

## Generate the full challenge file

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate a 5,000,000-day price series:

```bash
python coherent_ig_energy_submission.py --n 5000000 --seed 5 --out variance_timeseries.csv --summary-out full_submission_summary.csv
```

This creates `variance_timeseries.csv` with a single column named `Price`.

## Score with the public challenge scripts

Place `variance_timeseries.csv` in the directory where the public challenge scripts are run, then execute:

```bash
python data_loader_csv.py
python score_submission.py
```

If the scripts are inside a `code` folder:

```bash
python code/data_loader_csv.py
python code/score_submission.py
```

The included `variance_timeseries_100k.csv` is a short format check. A full-length run should be generated for challenge scoring.
