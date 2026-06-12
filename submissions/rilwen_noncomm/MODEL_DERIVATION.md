# An Ising-response noncommutative market-bath model

A market path can be read as the visible trace of a slower hidden environment: activity, pressure, liquidity, imbalance, and the order in which these forces arrive. The q-variance effect suggests that endpoint displacement and realised variance are not independent pieces of information. They share a hidden market state.

The construction below turns that idea into a path process. A daily market innovation is split into an even activity channel and an odd signed-pressure channel. The signed-pressure response follows the same two-state field response that appears in the one-site Ising calculation. The resulting activity-pressure signal is then passed through a persistent bath and corrected by order-sensitive terms.

The outcome is a stochastic process in which the endpoint move over a window and the realised variance inside that window are coupled through the same slowly decaying market state.

## 1. The object to be modelled

For a window of length $T$, q-variance describes the conditional realised variance inside the window as a function of the endpoint move $z$. The relevant empirical object is a surface,

$$Q(z,T) = \mathrm{E}[\mathrm{realised\ variance\ over\ }T \mid \mathrm{endpoint\ move}=z].$$

A static curve can describe the pooled relation $Q(z)$, but a path model must generate the whole family $Q(z,T)$. This requires a mechanism coupling endpoint displacement and realised variance inside the same window.

In an iid Gaussian path, the endpoint move and the realised variance are essentially independent. The model therefore introduces a persistent hidden market state. This hidden state controls activity over several days, so both endpoint displacement and realised variance are generated under the influence of the same bath.

## 2. Daily market innovation

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

## 3. Ising response for signed pressure

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

## 4. Baseline bath driver

The baseline driver combines unsigned activity and signed pressure:

$$\widetilde{D}_t = E_t - \tanh(\eta)O_t.$$

The standardised driver is

$$D_t = \mathrm{std}(\widetilde{D}_t).$$

The sign convention fixes the orientation of the signed-pressure channel. Reversing the convention for $O_t$ would reverse the sign.

## 5. Persistent bath memory

The hidden market state is not reset every day. It is a persistent memory of recent bath drivers.

Define

$$\rho = \exp\!\left(-\frac{1}{\mathrm{memory}}\right).$$

The unnormalised baseline bath evolves as

$$\widetilde{F}_{0,t} = \rho \widetilde{F}_{0,t-1} + \sqrt{1-\rho^2}\,D_{t-1}.$$

The baseline bath component is

$$F_{0,t} = \mathrm{std}(\widetilde{F}_{0,t}).$$

The one-day lag makes the bath causal: today's variance multiplier is driven by the previously accumulated market state, not by the same instantaneous innovation used in today's return.

The bath memory scale is the coherence scale of the hidden state. Larger memory means slower decay and more persistence across observation windows.

## 6. Order-sensitive market state

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

## 7. Persistent order-flow baths

The order-flow channels use the same memory scale as the baseline bath.

The antisymmetric bath is

$$\widetilde{F}_{C,t} = \rho \widetilde{F}_{C,t-1} + \sqrt{1-\rho^2}\,C_t.$$

Then

$$F_{C,t}=\mathrm{std}(\widetilde{F}_{C,t}).$$

The symmetric bath is

$$\widetilde{F}_{S,t} = \rho \widetilde{F}_{S,t-1} + \sqrt{1-\rho^2}\,S_t.$$

Then

$$F_{S,t}=\mathrm{std}(\widetilde{F}_{S,t}).$$

## 8. Orthogonalised bath components

The order-flow baths are orthogonalised so that they add new state information rather than simply rescaling the baseline bath.

For simulated series $X_t$ and $Y_t$, define the empirical inner product

$$\langle X,Y\rangle = \frac{1}{n}\sum_{t=1}^{n}X_tY_t.$$

The antisymmetric component after removing its projection on the baseline bath is

$$C_t^{\perp}=\mathrm{std}\!\left(F_{C,t}-\frac{\langle F_C,F_0\rangle}{\langle F_0,F_0\rangle}F_{0,t}\right).$$

The symmetric component is orthogonalised against both the baseline bath and the antisymmetric component:

$$S_t^{\perp}=\mathrm{std}\!\left(F_{S,t}-\frac{\langle F_S,F_0\rangle}{\langle F_0,F_0\rangle}F_{0,t}-\frac{\langle F_S,C^{\perp}\rangle}{\langle C^{\perp},C^{\perp}\rangle}C_t^{\perp}\right).$$

This is a deterministic Gram-Schmidt construction.

## 9. Hidden market bath

The hidden bath state is

$$F_t=\mathrm{std}\!\left(F_{0,t}+C_t^{\perp}-\tanh(\eta)S_t^{\perp}\right).$$

The antisymmetric component enters with unit weight. The symmetric correction enters with the same bounded signed-pressure response $\tanh(\eta)$ derived from the two-state imbalance model. The sign is part of the fixed model architecture.

## 10. Activity multiplier

The hidden bath controls daily market activity through

$$A_t=\frac{\exp(\eta F_t)}{\mathrm{E}[\exp(\eta F_t)]}.$$

The normalisation keeps average activity equal to one. The coupling $\eta$ sets the strength of bath-volatility coupling.

In the implementation, the exponential is evaluated in a numerically stable equivalent form by subtracting the maximum of $\eta F_t$ before normalisation. This does not change $A_t$, because the common factor cancels in the ratio.

## 11. Return process

The raw generated return is

$$r_t^{\ast}=\sqrt{\frac{\beta_{\mathrm{mult}}\sigma_0^2A_t}{252}}\,\varepsilon_t.$$

The simulated return path is demeaned:

$$r_t = r_t^{\ast} - \widehat{\mu}_{r^{\ast}}.$$

This imposes zero drift on the simulated path.

The price path is obtained from cumulative log returns. The absolute price level is irrelevant for q-variance, which depends on log returns. The code may rescale the price level by a constant for numerical safety; this leaves all returns unchanged.

## 12. Why q-variance appears

The q-variance surface relates endpoint displacement to realised variance. In this process, both are generated under the same persistent hidden state.

When the bath is elevated, daily variance is higher. A window that spends more time in an elevated bath state has higher realised variance. The same hidden state persists while the endpoint move is being generated. Endpoint displacement and realised variance are therefore not independent; they are coupled through the bath.

This is the dynamic mechanism behind q-variance in the model.

## 13. Relation to T-invariance

The bath coherence time is set by the bath memory scale. If the hidden state persists across the observation window, the q-variance relation remains comparatively stable across window lengths.

The model therefore supplies a concrete path process that generates a q-variance relation dynamically and gives a stable empirical fit across individual $T$-slices.
