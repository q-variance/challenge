# An Ising-response noncommutative market-bath model

A market path can be read as the visible trace of a slower hidden environment: activity, pressure, liquidity, imbalance, and the order in which these forces arrive. The q-variance effect suggests that endpoint displacement and realised variance are not independent pieces of information. They share a hidden market state.

The construction below turns that idea into a path process. A daily market innovation is split into an even activity channel and an odd signed-pressure channel. The signed-pressure response follows the same two-state field response that appears in the one-site Ising calculation. The resulting activity-pressure signal is then passed through a persistent bath and corrected by order-sensitive terms.

The outcome is a stochastic process in which the endpoint move over a window and the realised variance inside that window are coupled through the same slowly decaying market state.

## 1. The object to be modelled

For a window of length $T$, q-variance describes the conditional realised variance inside the window as a function of the endpoint move $z$. The relevant empirical object is a surface,

$$Q(z,T) = \mathrm{E}[\mathrm{realised\ variance\ over\ }T \mid \mathrm{endpoint\ move}=z].$$

A static curve can describe the pooled relation $Q(z)$, but a path model must generate the whole family $Q(z,T)$. This requires a mechanism coupling endpoint displacement and realised variance inside the same window.

In an iid Gaussian path, the endpoint move and the realised variance are essentially independent. The model therefore introduces a persistent hidden market state. This hidden state controls activity over several days, so both endpoint displacement and realised variance are generated under the influence of the same bath.


## 2. Continuous-time market-bath construction

Time is continuous. Let $X_t=\log P_t$ be the log price. The elementary signed market innovation is represented by a Brownian driver $W_t$. For any time cell $I_k=[t_k,t_k+\Delta t]$, define the standardised innovation

$$\varepsilon_k=\frac{W_{t_k+\Delta t}-W_{t_k}}{\sqrt{\Delta t}}.$$

Then

$$\varepsilon_k\sim N(0,1).$$

This construction has not chosen a daily time unit. It is defined for any cell length $\Delta t$.

### Activity and signed pressure from one Gaussian innovation

The signed-pressure coordinate is the signed innovation itself:

$$O_k=\varepsilon_k.$$

The variance/activity coordinate is the centred and normalised square of the same innovation:

$$E_k=\frac{\varepsilon_k^2-\mathbb{E}[\varepsilon_k^2]}{\sqrt{\mathrm{Var}(\varepsilon_k^2)}}.$$

For a standard Gaussian variable,

$$\mathbb{E}[\varepsilon_k^2]=1.$$

Also,

$$\mathrm{Var}(\varepsilon_k^2)=2.$$

Therefore

$$E_k=\frac{\varepsilon_k^2-1}{\sqrt{2}}.$$

The two coordinates are centred:

$$\mathbb{E}[O_k]=0,\qquad \mathbb{E}[E_k]=0.$$

They have unit variance:

$$\mathrm{Var}(O_k)=1,\qquad \mathrm{Var}(E_k)=1.$$

They are also orthogonal:

$$\mathrm{Cov}(O_k,E_k)=\mathbb{E}\left[\varepsilon_k\frac{\varepsilon_k^2-1}{\sqrt{2}}\right]=\frac{\mathbb{E}[\varepsilon_k^3]-\mathbb{E}[\varepsilon_k]}{\sqrt{2}}=0.$$

Thus the split into signed pressure and activity is obtained directly from the same Gaussian innovation. $O_k$ is the first signed coordinate. $E_k$ is the first variance coordinate. In Hermite language they are the first two non-constant Gaussian coordinates, but the construction does not depend on naming them that way.

In continuous-time notation these finite-cell coordinates are the cell projections of two orthogonal innovation fields: the signed field $O(t)$ and the centred quadratic-variation/activity field $E(t)$.

### Signed-pressure response

The signed-pressure response is bounded because a two-state pressure imbalance can only point in one of two directions. Let

$$s\in{-1,+1}.$$

Under an effective field $\eta$, the entropy-maximising probability weight with that field is

$$p(s)=\frac{\exp(\eta s)}{2\cosh(\eta)}.$$

The expected signed response is

$$m_\eta=\mathbb{E}[s]=\frac{\exp(\eta)-\exp(-\eta)}{\exp(\eta)+\exp(-\eta)}.$$

Hence

$$m_\eta=\tanh(\eta).$$

The activity-pressure driver is therefore

$$D(t)=\frac{E(t)-m_\eta O(t)}{\sqrt{1+m_\eta^2}}.$$

The denominator fixes the unit scale because $E(t)$ and $O(t)$ are orthogonal unit channels.

### Exponential bath memory

The market bath is a causal memory of past activity-pressure shocks. The memory kernel is

$$K_{\tau_B}(u)=\sqrt{\frac{2}{\tau_B}}\exp(-u/\tau_B)\mathbf{1}_{u\geq 0}.$$

The baseline bath is the causal convolution

$$F^0_t=\int_{-\infty}^{t}K_{\tau_B}(t-s)D(s),ds.$$

The parameter $\tau_B$ is the bath memory time. A disturbance $u$ units in the past is weighted by

$$\exp(-u/\tau_B).$$

After one memory time $\tau_B$, its weight is reduced by $e^{-1}$.

### Ordered activity-pressure interaction

The bath is not only sensitive to the amount of activity and signed pressure. It is also sensitive to their order.

The filtered activity state is

$$U_t=\int_{-\infty}^{t}K_{\tau_B}(t-s)E(s),ds.$$

The filtered signed-pressure state is

$$V_t=\int_{-\infty}^{t}K_{\tau_B}(t-s)O(s),ds.$$

Their causal histories under the same bath memory are

$$\bar U_t=\int_{-\infty}^{t}K_{\tau_B}(t-s)U_s,ds.$$

and

$$\bar V_t=\int_{-\infty}^{t}K_{\tau_B}(t-s)V_s,ds.$$

There are two bilinear causal orderings:

$$U_t\bar V_t.$$

and

$$V_t\bar U_t.$$

The first term corresponds to recent activity acting on older signed-pressure history. The second corresponds to recent signed pressure acting on older activity history.

The antisymmetric ordered component is

$$C_t=U_t\bar V_t-V_t\bar U_t.$$

The symmetric ordered component is

$$S_t=U_t\bar V_t+V_t\bar U_t.$$

The antisymmetric component changes sign when the order of activity and signed pressure is reversed. The symmetric component measures the corresponding order-flow intensity.

Both components use the same memory time $\tau_B$. No additional memory scale is introduced.

### Orthogonal bath state

For any stationary process $Y_t$, define

$$\mathcal{N}[Y_t]=\frac{Y_t-\mathbb{E}[Y_t]}{\sqrt{\mathrm{Var}(Y_t)}}.$$

The baseline bath mode is

$$B^0_t=\mathcal{N}[F^0_t].$$

The antisymmetric mode after removing its projection on the baseline bath is

$$B^C_t=\mathcal{N}\left[C_t-\frac{\mathrm{Cov}(C_t,B^0_t)}{\mathrm{Var}(B^0_t)}B^0_t\right].$$

The symmetric mode after removing its projections on the previous bath modes is

$$B^S_t=\mathcal{N}\left[S_t-\frac{\mathrm{Cov}(S_t,B^0_t)}{\mathrm{Var}(B^0_t)}B^0_t-\frac{\mathrm{Cov}(S_t,B^C_t)}{\mathrm{Var}(B^C_t)}B^C_t\right].$$

This is Gram-Schmidt orthogonalisation. It fixes the geometry of the bath modes and introduces no fitted coefficient.

The hidden market bath is

$$F_t=\mathcal{N}\left[B^0_t+B^C_t-m_\eta B^S_t\right].$$

The same bounded signed-pressure response $m_\eta=\tanh(\eta)$ controls the symmetric correction.

### Volatility generated by the bath

The hidden bath controls local variance through a mean-one positive activity multiplier:

$$A_t=\frac{\exp(\eta F_t)}{\mathbb{E}[\exp(\eta F_t)]}.$$

Therefore

$$A_t>0.$$

Also,

$$\mathbb{E}[A_t]=1.$$

Let $v_0$ be the variance scale. The continuous-time price law is

$$dX_t=\sqrt{v_0 A_t},dW_t.$$

Thus the conditional instantaneous variance is

$$\mathrm{Var}(dX_t\mid F_t)=v_0 A_t,dt.$$

The three independent controls are

$$v_0,\qquad \tau_B,\qquad \eta.$$

They correspond to variance scale, bath memory time, and bath-volatility coupling.

### Observation on a daily grid

The continuous-time construction determines its sampled form on any observation grid. For a grid spacing $\Delta t$, the exponential memory coefficient is

$$\rho_{\Delta t}=\exp(-\Delta t/\tau_B).$$

If the unit of time is one trading day and $\Delta t=1$, then

$$\rho=\exp(-1/\tau_B).$$

The daily variables

$$O_k=\varepsilon_k,\qquad E_k=\frac{\varepsilon_k^2-1}{\sqrt{2}}$$

are the finite-cell signed-pressure and activity coordinates derived above.

On a daily grid, the earliest non-anticipating cell products that distinguish the two orderings are

$$E_{k-1}O_{k-2}.$$

and

$$O_{k-1}E_{k-2}.$$

Their difference is the daily-grid antisymmetric ordered component, and their sum is the daily-grid symmetric ordered component. These are not fitted lags. They are the first causal daily representatives of the two continuous ordered products $U_t\bar V_t$ and $V_t\bar U_t$.

The implementation uses daily cell variables because the benchmark price paths, empirical q-variance data, and scorer are daily. The model object itself is the continuous-time market bath $F_t$.



## 3. Daily market innovation

Let

$$\varepsilon_t \sim N(0,1)$$

be the new unpredictable market innovation arriving on day $t$. The innovation is decomposed into an even activity channel and an odd signed-pressure channel.

The even activity channel is

$$E_t = \frac{\varepsilon_t^2 - 1}{\sqrt{2}}.$$

It is large on unusually active days, regardless of whether the price move is positive or negative.

The odd signed-pressure channel is

$$O_t = \varepsilon_t.$$

It keeps the sign of the pressure.

For any simulated path variable $X_t$, write

$$\mathrm{std}(X_t) = \frac{X_t-\widehat{\mu}_X}{\widehat{\sigma}_X}$$

for finite-sample standardisation to zero mean and unit variance. This is a deterministic path normalisation.

## 4. Ising response for signed pressure

The market pressure channel is represented at the fast level by a two-state signed variable

$$s_t \in \{-1,+1\}.$$

The state $+1$ corresponds to buy-pressure dominance and $-1$ corresponds to sell-pressure dominance. The coupling $\eta$ acts as an effective directional field on this binary pressure state.

Let

$$p_+ = \mathrm{P}(s_t=+1), \qquad p_- = \mathrm{P}(s_t=-1).$$

Use the maximum-entropy distribution biased by the effective field. The entropy is

$$H(p_+,p_-) = -p_+\log p_+ - p_-\log p_-.$$

The normalisation constraint is

$$p_+ + p_- = 1.$$

The field contribution is

$$\eta\,\mathrm{E}[s_t] = \eta(p_+ - p_-).$$

The Lagrangian is therefore

$$\Phi = -p_+\log p_+ - p_-\log p_- + \lambda(p_+ + p_- - 1) + \eta(p_+ - p_-).$$

For $s\in\{-1,+1\}$, this can be written as

$$\Phi = -\sum_s p_s \log p_s + \lambda\left(\sum_s p_s - 1\right) + \eta\sum_s s\,p_s.$$

Taking derivatives gives

$$\frac{\partial\Phi}{\partial p_s} = -\log p_s - 1 + \lambda + \eta s = 0.$$

Hence

$$p_s = \exp(\lambda-1)\exp(\eta s).$$

Normalisation gives

$$Z(\eta)=\exp(\eta)+\exp(-\eta)=2\cosh(\eta).$$

Therefore

$$p(s_t=s)=\frac{\exp(\eta s)}{2\cosh(\eta)}, \qquad s\in\{-1,+1\}.$$

The expected normalised signed imbalance is

$$m(\eta)=\mathrm{E}[s_t].$$

Substituting the two probabilities,

$$m(\eta)=\frac{\exp(\eta)-\exp(-\eta)}{\exp(\eta)+\exp(-\eta)}.$$

Thus

$$m(\eta)=\tanh(\eta).$$

This is the one-site Ising response: an effective field is converted into a bounded order-parameter response. In this model, the order parameter is not magnetisation; it is the relative signed-pressure contribution in the market bath.

## 5. Baseline bath driver

The baseline driver combines unsigned activity and signed pressure:

$$\widetilde{D}_t = E_t - \tanh(\eta)O_t.$$

The standardised driver is

$$D_t = \mathrm{std}(\widetilde{D}_t).$$

The sign convention fixes the orientation of the signed-pressure channel. Reversing the convention for $O_t$ would reverse the sign.

## 6. Persistent bath memory

The hidden market state is not reset every day. It is a persistent memory of recent bath drivers.

Define

$$\rho = \exp\!\left(-\frac{1}{\mathrm{memory}}\right).$$

The unnormalised baseline bath evolves as

$$\widetilde{F}_{0,t} = \rho \widetilde{F}_{0,t-1} + \sqrt{1-\rho^2}\,D_{t-1}.$$

The baseline bath component is

$$F_{0,t} = \mathrm{std}(\widetilde{F}_{0,t}).$$

The one-day lag makes the bath causal: today's variance multiplier is driven by the previously accumulated market state, not by the same instantaneous innovation used in today's return.

The bath memory scale is the coherence scale of the hidden state. Larger memory means slower decay and more persistence across observation windows.

## 7. Order-sensitive market state

The baseline bath remembers recent activity and pressure. The next step is to make the state sensitive to the order in which activity and signed pressure arrive.

Define the two ordered products

$$a_t = E_{t-1}O_{t-2}, \qquad b_t = O_{t-1}E_{t-2}.$$

The antisymmetric order-flow channel is

$$\widetilde{C}_t = a_t - b_t.$$

The symmetric order-flow channel is

$$\widetilde{S}_t = a_t + b_t.$$

The antisymmetric term changes sign when the order of activity and signed pressure is reversed. It is the commutator-like part of the model. The symmetric term is the corresponding order-flow intensity component.

Both channels are standardised:

$$C_t = \mathrm{std}(\widetilde{C}_t), \qquad S_t = \mathrm{std}(\widetilde{S}_t).$$

## 8. Persistent order-flow baths

The order-flow channels use the same memory scale as the baseline bath.

The antisymmetric bath is

$$\widetilde{F}_{C,t} = \rho \widetilde{F}_{C,t-1} + \sqrt{1-\rho^2}\,C_t.$$

Then

$$F_{C,t}=\mathrm{std}(\widetilde{F}_{C,t}).$$

The symmetric bath is

$$\widetilde{F}_{S,t} = \rho \widetilde{F}_{S,t-1} + \sqrt{1-\rho^2}\,S_t.$$

Then

$$F_{S,t}=\mathrm{std}(\widetilde{F}_{S,t}).$$

## 9. Orthogonalised bath components

The order-flow baths are orthogonalised so that they add new state information rather than simply rescaling the baseline bath.

For simulated series $X_t$ and $Y_t$, define the empirical inner product

$$\langle X,Y\rangle = \frac{1}{n}\sum_{t=1}^{n}X_tY_t.$$

The antisymmetric component after removing its projection on the baseline bath is

$$C_t^{\perp}=\mathrm{std}\!\left(F_{C,t}-\frac{\langle F_C,F_0\rangle}{\langle F_0,F_0\rangle}F_{0,t}\right).$$

The symmetric component is orthogonalised against both the baseline bath and the antisymmetric component:

$$S_t^{\perp}=\mathrm{std}\!\left(F_{S,t}-\frac{\langle F_S,F_0\rangle}{\langle F_0,F_0\rangle}F_{0,t}-\frac{\langle F_S,C^{\perp}\rangle}{\langle C^{\perp},C^{\perp}\rangle}C_t^{\perp}\right).$$

This is a deterministic Gram-Schmidt construction.

## 10. Hidden market bath

The hidden bath state is

$$F_t=\mathrm{std}\!\left(F_{0,t}+C_t^{\perp}-\tanh(\eta)S_t^{\perp}\right).$$

The antisymmetric component enters with unit weight. The symmetric correction enters with the same bounded signed-pressure response $\tanh(\eta)$ derived from the two-state imbalance model. The sign is part of the fixed model architecture.

## 11. Activity multiplier

The hidden bath controls daily market activity through

$$A_t=\frac{\exp(\eta F_t)}{\mathrm{E}[\exp(\eta F_t)]}.$$

The normalisation keeps average activity equal to one. The coupling $\eta$ sets the strength of bath-volatility coupling.

In the implementation, the exponential is evaluated in a numerically stable equivalent form by subtracting the maximum of $\eta F_t$ before normalisation. This does not change $A_t$, because the common factor cancels in the ratio.

## 12. Return process

The raw generated return is

$$r_t^{\ast}=\sqrt{\frac{\beta_{\mathrm{mult}}\sigma_0^2A_t}{252}}\,\varepsilon_t.$$

The simulated return path is demeaned:

$$r_t = r_t^{\ast} - \widehat{\mu}_{r^{\ast}}.$$

This imposes zero drift on the simulated path.

The price path is obtained from cumulative log returns. The absolute price level is irrelevant for q-variance, which depends on log returns. The code may rescale the price level by a constant for numerical safety; this leaves all returns unchanged.

## 13. Why q-variance appears

The q-variance surface relates endpoint displacement to realised variance. In this process, both are generated under the same persistent hidden state.

When the bath is elevated, daily variance is higher. A window that spends more time in an elevated bath state has higher realised variance. The same hidden state persists while the endpoint move is being generated. Endpoint displacement and realised variance are therefore not independent; they are coupled through the bath.

This is the dynamic mechanism behind q-variance in the model.

## 14. Relation to T-invariance

The bath coherence time is set by the bath memory scale. If the hidden state persists across the observation window, the q-variance relation remains comparatively stable across window lengths.

The model therefore supplies a concrete path process that generates a q-variance relation dynamically and gives a stable empirical fit across individual $T$-slices.



