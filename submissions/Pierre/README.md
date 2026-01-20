
Team: Pierre

==========================================================


Model description


The model is described by the following SDEs:

Price:

 d(ln(P_t)) = sqrt(V_t) dW_{1,t},

where the shock dW_{1,t} is a standard Wiener process.

Variance:

V_t = Phi(z_t) * exp(\zeta dW_{2,t} - 1/2 \zeta^2 dt),

where zeta it the vol-of-vol, and Phi(z_t) is the q-variance target:

Phi(z_t) = \sigma_0^2 + 0.5 * (z_t - z_0)^2 - Shift,

and Shift a numerical correction equal to 0.025.

==========================================================

The three free parameters are

 \sigma_0
z_0
\zeta

==========================================================

The Shift counteracts the Ito Bias that shifts the curve upwards.

==========================================================

Files

- ‘dataset.parquet’: columns ‘ticker’, ‘date’, ’T’, ‘z’, ‘sigma’

- ‘simulated_prices_logreturns_100k.csv’: 100,000 daily prices (‘Price’ column)

- ‘Model_q-variance.py’: generates prices and returns

==========================================================