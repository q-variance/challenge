Team

AURA

Description

We model log-prices via a gated semimartingale: outside repricing the process follows a càdlàg semimartingale $U$, while during predictable repricing windows the underlying is 
switched off and the log-price evolves by a finite-variation execution path. 
For a single repricing event executed at constant rate and observed in two steps, realized variance satisfies $RV=\tfrac12 R^2$, implying $\widehat V=RV/T=\tfrac12 z^2$ and 
hence a quadratic $q$-variance coefficient $q=0.5$. 
Equal splitting (and thus $q=0.5$) may also arise from trading-time / time-zone or calendar timestamping, and should therefore be assessed separately from true execution mechanics.

Parameters

1)  sigma_base: baseline diffusion multiplier on non-execution days.
2)  p_exec: per-day probability to start a 2-day execution block (subject to cooldown).
3)  U_scale: execution move scale in units of daily volatility.

cooldown: structural refractory gap (days) enforcing non-overlapping 2-day execution blocks; set to 0 if not needed.
   If an event starts on day t and executes on days t and t+1, then no new event may start before day (t + 2 + cooldown).

mu_ann: drift; irrelevant here and can be set to 0.
sigma_ann: fixed by the challenge.


Note

The mechanism is model-agnostic: it does not prescribe the baseline price dynamics (diffusion, jumps, stochastic volatility, etc.). The q-variance effect is induced solely by the two-step execution/observation (temporal coarsening) and can be superimposed on an arbitrary baseline model.

One could further parameterize p_exec and U_scale as functions of sigma_base (e.g., to keep event frequency/impact comparable across baseline noise levels), but this is outside the scope of this submission.

