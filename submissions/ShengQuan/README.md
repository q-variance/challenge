# Jump-Diffusion Model with Volatility Resampling

We describe the **jump-diffusion model with volatility resampling** in the following:

$$
  dS_t = \delta (dN_t - \lambda dt) + \begin{cases}
      \sqrt{v_0} dW_t, & 0\le t < \tau_1\\
      \sqrt{v_1} dW_t, & \tau_1 \le t < \tau_2\\
      \sqrt{v_2} dW_t, & \tau_2 \le t < \tau_3\\
      \quad \vdots \\
      \sqrt{v_{N_t}} dW_t, & t \ge \tau_{N_t},
      \end{cases}
$$

where $\tau_1 < \tau_2 < \cdots < \tau_{N_t}$ are the successive arrival times of a Poisson process $N_t$. In it, there are three (3) sources of randomness:

- the volatility resampling from an inverse gamma distribution;
- the Brownian motion driving the diffusion component of the price dynamics; and
- the Poisson process that triggers the volatility resampling and the jumps,

as well as four (4) free parameters:

- inverse gamma distribution shape parameter $3/2$;
- inverse gamma distribution scale parameter $v_0$;
- the jump amplitude $\delta$; and
- the Poisson process arrival rate $\lambda$.

The shape parameter of the inverse gamma distribution is set to $\frac{3}{2}$ to yield the $\frac{1}{2}$ quadratic coefficient of the q-variance; the scale parameter $v_0$ of the inverse gamma distribution is directly related to the minimum volatility $\sigma_0$ of the q-variance: $v_0\approx \sigma_0^2$; the jump amplitude $\delta$ is a mechanism to generate skewness and a handle to the q-variance offset $z_0$; finally, the Poisson $\lambda$ regulates a slow volatility regime change so that the volatility persists over a time horizon over 26 weeks.

The model construction and the numerical results are detailed in the attached document. The description is fully transparent and straightforward to implement. No code or simulated data is supplied at this time, pending internal compliance review.
