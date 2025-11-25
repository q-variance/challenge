# Q-Variance Challenge

<img src="Figure_1.png" width="800">

Can any continuous-time stochastic-volatility model reproduce the parabolic relationship  
σ(z)² = σ₀² + (z-zoff)²/2
across all horizons 1-26 weeks with R² ≥ 0.995 and ≤ 2 free parameters?

Here z = x/sqrt(T) where x is the log price change over a period T, adjusted for drift. Blue points in the figure are variance vs z for stocks from the S&P 500. Blue line is average variance as a function of z, red line is the q-variance curve. Read the [Q-Variance Wilmott paper](Q-Variance_Wilmott_July2025.pdf) for more details.

Repository contains:
- Data set (dataset.parquet) in three parts containing price data for 352 stocks from the S&P 500 (stocks with less than 25 years of data excluded)
- Full dataset generator (data_loader.py) to show how the data was generated
- Scoring engine
- Live leaderboard
- Baseline quantum fit
- Plot Figure_1.png showing q-variance and R² value for the actual data

For example, to try a rough vol model, simulate a long price series, compute sigma²(z) for each window, output new Parquet. You can also do multiple simulations: assign each a different ticker and the code will average over them as if they are different stocks.

Dataset: 352 S&P 500 stocks (>25 year history), 1–26 weeks T, ~300K rows. 

Columns: ticker (str), date (date), T (int), sigma (float, annualized vol), z (float, scaled log return).

Due to file size limitations, the dataset parquet file is divided into three parts. Combine them with the command:
```python
df = pd.concat([pd.read_parquet("dataset_part1.parquet"),pd.read_parquet("dataset_part2.parquet"),pd.read_parquet("dataset_part3.parquet")])
```

Python dependencies: pip install yfinance pandas numpy scipy matplotlib pyarrow

## Scoring the Challenge

The challenge scores submissions on **one global R²** over the **entire dataset**. Since the quantum model with σ₀=0.255 and zoff = 0.02 gives a near-perfect fit (R² = 0.998) this curve can be used as a proxy for the real data. In other words, the aim is to fit the parabola.

### How Scoring Works
1. **Load Submission**: `score_submission.py` reads your `dataset.parquet` (must match format: ticker, date, T, z, sigma).
2. **Compute Variance**: Converts sigma → var = sigma².
3. **Global Binning**: Bins z from -0.6 to 0.6 (delz=0.025), averages var per bin (as in `baseline_fit.py` global plot).
4. **Fit**: Fits var = σ₀² + (z-zoff)²/2 to binned averages, computes R².
5. **Threshold**: R² ≥ 0.995 with no more than two free parameters wins the challenge (agreement of quantum with data is 0.998).

### Test Your Submission
Run the test mode to score your Parquet:

```bash
python3 score_submission.py
```

1. Fork this repository
2. Place your model output in `submissions/your_team_name/` as:
   - `dataset.parquet` (must have columns: ticker, date, T, z, sigma)
3. Add a `README.md` in your folder with:
   - Team name
   - Short model description
   - Contact (optional)
4. Open a Pull Request titled: "Submission: [Your Team Name]"

**Frequently Asked Questions**

Q: What is q-variance – is it a well-known "stylized fact"?

A: No, a stylized fact is just a general observation about market data. Q-variance is a falsifiable prediction because the multiplicative constant on the quadratic term is not a fit, it is set by theory at 0.5. The same formula applies for all period lengths T.

Q: Is q-variance a large effect?

A: Yes, the minimum variance is about half the total variance so this is a large effect. If you are modelling variance then you need to take q-variance into account.

Q: Has q-variance been previously reported in the literature?

A: Not to our knowledge, and we have asked many experts, but please bring any references to our attention. If anyone has made the exact same prediction using a model then we will announce them the winner.

Q: Does q-variance have implications for quantitative finance?

A: Yes, classical finance assumes a diffusive model for price change, which does not produce q-variance. Standard formulas such as Black-Scholes or the formula used to calculate VIX will therefore not work as expected.

Q: Is q-variance related to the implied volatility smile?

A: Yes, however it is not the same thing because q-variance applies to realized volatility. But if you want to model implied volatility, a first step is to understand realized volatility.

Q: Is q-variance related to the price-change distribution over a period?

A: Yes, it implies that price-change follows the q-distribution which is a particular time-invariant, Poisson-weighted sum of Gaussians (see further reading below). [Figure 4](Figure_4.png) compares the q-distribution with the average distribution over the S&P 500 stocks, where the distribution of each stock has been normalized by its standard deviation for comparability.

Q: How does q-variance vary over different time periods, or from stock to stock?

A: In theory the curve should be time-invariant, though in practice there is a small degree of variation, see [Figure 2](Figure_2.png). The results for individual stocks are of course noisier as shown in [Figure 3](Figure_3.png), but taking the average variance over a number of stocks smooths out this noise.

Q: What is the point in using a classical model if the quantum model is an almost perfect match to the data?

A: The quantum model predicts variance and the price-change distribution, but does not provide a time series of daily prices. If a classical model can do that, and still produce the quadratic shape, then that will be very useful. We will therefore also give an honorable mention to any classical entry which can come close to matching the quantum model even if it involves extra parameters (though parsimony is important in order to understand how the model works, otherwise it is just a fitting exercise).

Q. What does this have to do with subatomic particles?

A. Nothing, other than the fact that some problems which couple probability and dynamics in both physics and finance apparently benefit from using a type of probability that is based on complex numbers. And sometimes you have to trust the mathematics.

Q: Can I use AI for the challenge?

A: Yes, AI-assisted entries are encouraged. We used Grok to help design and code the challenge. Its [entry](submissions/grok_rough_vol) is a modified rough volatility model which uses four parameters to achieve an R² of 0.986 – close but no cigar! The aim is to find a process which can achieve better results in a more parsimonious fashion.

**Further reading:**

Orrell D (2025) A Quantum Jump Model of Option Pricing. The Journal of Derivatives 33(2).

Orrell D (2025) Quantum impact and the supply-demand curve. Philosophical Transactions of the Royal Society A 383(20240562).

Orrell D (2026) The Quantum Stock Market. MIT Press (in press).


