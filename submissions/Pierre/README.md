
Team: Pierre

==========================================================


Model description


The model is described by the following SDEs:

Price:

 d(ln(P_t)) = sqrt(V_t) dW{1,t},

where the shock dW_{1,t} is a standard Wiener process.

Variance:

dV_t = kappa (Phi(z_t) - V_t) dt + \zeta \sqrt{V_t} dW_{2,t},

where kappa is a mean reversion speed, zeta it the vol-of-vol, and Phi(z_t) is the q-variance target:

Phi(z_t) = \sigma_0^2 + 0.5 * (z_t - z_0)^2 - Shift,

and Shift a numerical correction equal to 0.025.


==========================================================

The three free parameters are

 \sigma_0
z_0
\zeta

==========================================================

The parameter \kappa is a numerical convergence factor due to discrete calibration steps.

The Shift counteracts the Ito Bias that shifts the curve upwards.

==========================================================

Files

- ‘dataset.parquet’: columns ‘ticker’, ‘date’, ’T’, ‘z’, ‘sigma’

- ‘simulated_prices_logreturns_100k.csv’: 100,000 daily prices (‘Price’ column)

- ‘Model_q-variance.py’: generates prices and returns

==========================================================