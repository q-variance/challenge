# Q-Variance Challenge
![](Figure_1.png)

Can any continuous-time stochastic-volatility model reproduce the parabolic relationship  
σ(z) = √(σ₀² + z²/2)  
across 352 assets and all horizons 1-26 weeks with R² ≥ 0.99 and ≤ 2 free parameters?

Here z = x/sqrt(T) where x is the log price change over a period T, adjusted for drift. Read the paper Q-Variance_Wilmott_July2025.pdf for more details.

Quantum baseline (Orrell 2025): R² ≈ 0.998. Uses two parameters, σ₀ and a small horizontal offset

Repository contains:
- Data set (dataset.parquet) containing price data for 352 stocks from the S&P 500 (stocks with less than 25 years of data were not included)
- Full dataset generator (data_loader.py) to show how the data was generated
- Scoring engine
- Live leaderboard
- Baseline quantum fit
- Plot Figure_1.png showing q-variance and R^2 value
- Plot Figure_2.png showing the first 100 stocks.

For example, to try a rough vol model, simulate paths with dV_t = -κ V_t dt + σ V_t^γ dW_t (γ=0.1), compute sigma(z) for each window, output new Parquet.

Dataset: 352 S&P 500 stocks (1992+ history), 1–26 weeks T, ~300K rows. 

Columns: ticker (str), date (date), T (int), sigma (float, annualized vol), z (float, scaled log return).

Due to file size limits, the parquet file is divided into three parts. Combine them with the command:
```python
df = pd.concat([pd.read_parquet("dataset_part1.parquet"),pd.read_parquet("dataset_part2.parquet"),pd.read_parquet("dataset_part3.parquet")])
```

Python dependencies: pip install yfinance pandas numpy scipy matplotlib pyarrow


## Scoring the Challenge

The challenge scores submissions on **one global R²** over the **entire dataset** (all 352 stocks, all 26 horizons).

### How Scoring Works
1. **Load Submission**: `score_submission.py` reads your `dataset.parquet` (must match format: ticker, date, T, z, sigma).
2. **Compute Variance**: Converts sigma → var = sigma².
3. **Global Binning**: Bins z from -0.6 to 0.6 (delz=0.025), averages var per bin (as in `baseline_fit.py` global plot).
4. **Fit**: Fits var = σ₀² + z²/2 to binned averages, computes R².
5. **Threshold**: R² ≥ 0.95 wins the challenge (baseline: 0.99).

### Test Your Submission
Run the test mode to score your Parquet:

```bash
python3 score_submission.py
```

1. Fork this repository
2. Place your model output in `submissions/your_team_name/` as:
   - `prize_dataset.parquet` (must have columns: ticker, date, T, z, sigma)
3. Add a `README.md` in your folder with:
   - Team name
   - Short model description
   - Contact (optional)
4. Open a Pull Request titled: "Submission: [Your Team Name]"

GitHub Actions will automatically run `score_submission.py` and post your score.

**Challenge Rules**
- Must use **variance** (sigma²)
- Must cover **all 352 stocks** and **T = 5,10,...,130 days**

Good luck.


**Frequently Asked Questions**

Q: Is q-variance a well-known "stylized fact"?

A: No, a stylized fact is just a general observation about market data, as opposed to a firm prediction. Q-variance is a falsifiable prediction because the multiplicative constant on the quadratic term is not a fit, it is set by theory at 0.5. The same formula applies for all period lengths T.

Q: Is q-variance a large effect?

A: Yes, the minimum variance is about half the total variance so this is a large effect. If you are modelling variance then you need to take q-variance into account.

Q: Has q-variance been previously reported in the literature?

A: Not to our knowledge, and we have asked many experts, but please bring any references to our attention.

Q: Does q-variance have implications for finance?

A: Yes, it means that standard formulas such as Black-Scholes or the formula used to calculate VIX will not work as expected.

Q: Is q-variance related to the implied volatility smile?

A: Yes, but it is not the same thing because q-variance applies to realized volatility.

Q: Is q-variance related to the price-change distribution over a period?

A: Yes, it implies that price-change follows the q-distribution which is a particular time-invariant, Poisson-weighted sum of Gaussians.

Q: Can I use AI for the challenge?

A: Yes, AI entries are encouraged. Try this prompt: "Fork this repo and try a rough vol model to beat R² = 0.99."

**Further reading:**

Orrell D (2025) A Quantum Jump Model of Option Pricing. The Journal of Derivatives 33(2).

Orrell D (2025) Quantum impact and the supply-demand curve. Philosophical Transactions of the Royal Society A 383(20240562).
