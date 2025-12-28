# Summary of submissions as of end-2025

We have now had a number of excellent entries to the competition (not all of whom posted to github) and wish to thank all the participants. This note summarises again the question, and the answers received so far. Although the competition was originally framed as an end-of-year challenge, it remains open.

Q-variance is all about the coefficient of $z^2$. The model showed it must be 0.5. Some common models may show a quadratic, and some may even show the coefficient is independent of period $T$. But that coefficient is a parameter in the models, it can be chosen (for example by adjusting a shape factor). None have to have 0.5. 

This competition goes further, by asking – not if a continuous-time model can predict q-variance – but whether any such model can even **produce** q-variance, in a reliable fashion, using up to three parameters: a base volatility, a horizontal offset, and an extra parameter. The last might be something that is fit to data, or it could be a factor which is selected with the sole justification of matching q-variance.

Four submissions drew on the idea of sampling variance from an inverse-gamma distribution with shape factor set to $\alpha = 3/2$ and rate equal to a base variance $\sigma^2$. While this distribution does agree in principle with q-variance, you still have to find a time series which matches it over all periods $T$. 

[One entry](https://github.com/q-variance/challenge/tree/main/submissions/simu.ai) (since withdrawn) tackled this in an ingenious way by adding an extra layer of regime switching, however this put the number of tuneable parameters over the limit. It also couldn’t address the other problem, which is that the inverse-gamma distribution is extremely noisy, with huge spikes, and the variance is undefined. The result is that the model (and its $R^2$ score) only converges over simulations of thousands of years, and even then is sensitive to whether the selected time period contains a certain spike. 

Other entries have used stochastic volatility or GARCH approaches, however the models either introduced extra parameters or did not agree with q-variance to the required standard (although it is reassuring to see that an optimized [GARCH model](https://github.com/q-variance/challenge/tree/main/submissions/tingjun2) can do a good job of modelling q-variance). We therefore have no firm winner, but a number of close entries (Wilmott magazine to announce later).

Happy New Year!
