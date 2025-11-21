# Q-Variance Challenge

Can any continuous-time stochastic-volatility model reproduce the parabolic relationship  
σ(z) = √(σ₀² + z²/2)  
across 8 assets and all horizons 1-26 weeks with R² ≥ 0.92 and ≤ 2 free parameters?

Here z = x/sqrt(T) where x is the log price change over a period T, adjusted for drift.
Read the paper Q-Variance_Wilmott_July2025.pdf for more details.

Quantum baseline (Orrell 2025): R² ≈ 0.998

Repository contains:
- Full dataset generator (data_loader.py)
- Scoring engine
- Live leaderboard
- Baseline quantum fit

To test your model, download the file prize_dataset.parquet which contains price data for 352 stocks from the S&P 500 (stocks with less than 25 years of data were not included). To see how the file was generated, or to generate it yourself, see data_loader.py. Then use your model to see if you can fit the plot of binned variance versus z using no more than two parameters per stock (the model must work for all period lengths T). Figure_1.png is a plot showing the q-variance and R^2 value, Figure_2.png shows the first 100 stocks.

Frequently Asked Questions

Q: Is q-variance a well-known "stylized fact"?
A: No, a stylized fact is just a general observation about market data, as opposed to a firm prediction. Q-variance is a falsifiable prediction because the multiplicative constant on the quadratic term is not a fit, it is set by theory at 0.5. The same formula applies for all period lengths T.

Q: Is q-variance a large effect?
A: Yes, the minimum variance is about half the total variance so this is a large effect.

Q: Has q-variance been previously reported in the literature?
A: Not to our knowledge, and we have asked many experts, but please bring any references to our attention.

Q: Does q-variance have implications for finance?
A: Yes, it means that standard formulas such as Black-Scholes or the formula used to calculate VIX will not work as expected.

Q: Is q-variance related to the implied volatility smile?
A: Yes, but it is not the same thing because it applies to realized volatility.

Q: Can I use AI for the challenge?
A: Yes, AI entries are encouraged.
