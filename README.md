# Q-Variance Challenge

<img src="Figure_1.png" width="800">

Can any continuous-time stochastic-volatility model reproduce the parabolic relationship  
σ(z)² = σ₀² + (z-zoff)²/2
across all horizons 1-26 weeks with R² ≥ 0.995 and ≤ 3 free parameters?

Here z = x/sqrt(T) where x is the log price change over a period T, adjusted for drift. Blue points in the figure are variance vs z for stocks from the S&P 500. Blue line is average variance as a function of z, red line is the q-variance curve. Read the [Q-Variance Wilmott paper](Q-Variance_Wilmott_July2025.pdf) for more details.

Repository contains:
- Parquet file in three parts containing price data for 352 stocks from the S&P 500 (stocks with less than 25 years of data excluded)
- Full dataset generator [data_loader.py](data_loader.py) to show how the data was generated
- Scoring engine
- Baseline model fit
- Plot [Figure 1](Figure_1.png) showing q-variance and R² value for the actual data
- Jupyter notebook [qvariance_single.ipynb](notebooks/qvariance_single.ipynb) showing how to compute q-variance for a single asset

For example, to try a rough vol model, simulate a long price series, compute sigma²(z) for each window, output new parquet. You can also do multiple simulations: assign each a different ticker and the code will average over them as if they are different stocks.

Dataset: 352 stocks from the S&P 500 (>25 year history), 1–26 weeks T, ~300K rows. 

Columns: ticker (str), date (date), T (int), sigma (float, annualized vol), z (float, scaled log return).

Due to file size limitations, the dataset parquet file is divided into three parts. Combine them with the command:
```python
df = pd.concat([pd.read_parquet("dataset_part1.parquet"),pd.read_parquet("dataset_part2.parquet"),pd.read_parquet("dataset_part3.parquet")])
```

Python dependencies: pip install yfinance pandas numpy scipy matplotlib pyarrow

## Scoring the Challenge

The challenge scores submissions on **one global R²** over the **entire dataset**. Since the q-variance parabola with σ₀=0.255 and zoff = 0.02 gives a near-perfect fit (R² = 0.998) this curve can be used as a proxy for the real data. In other words, the aim is to fit the two-parameter parabola, using **up to three parameters** – must be easy, right?

### How Scoring Works
1. **Load Submission**: `score_submission.py` reads your `dataset.parquet` (must match format: ticker, date, T, z, sigma).
2. **Compute Variance**: Converts sigma → var = sigma².
3. **Global Binning**: Bins z from -0.6 to 0.6 (delz=0.025), averages var per bin (as in `baseline_fit.py` global plot).
4. **Fit**: Fits var = σ₀² + (z-zoff)²/2 to binned averages, computes R².
5. **Threshold**: R² ≥ 0.995 with no more than three free parameters (agreement of parabola with data is 0.998). The price-change distribution in z should also be time-invariant, so the model should be independent of period length T.

### Test Your Submission
Run the test mode to score your simulation data:

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

## Frequently Asked Questions

Q: Why should I enter this competition?

A: For fun, an intellectual challenge, kudos. But also because, if your existing model of price behaviour doesn't do q-variance, then it is missing important market structure.

Q: Is q-variance a well-known "stylized fact"?

A: No, a stylized fact is just a general observation about market data. Q-variance is a **falsifiable prediction** because the multiplicative constant on the quadratic term is not a fit, it is set by theory at 0.5. The same formula applies for all period lengths T. As far as we are aware this is the most clear-cut and easily tested example of a model prediction in finance.

Q: Is q-variance a large effect?

A: Yes, the minimum variance is about half the total variance so this is a large effect. If you are modelling variance then you need to take q-variance into account.

Q: Has q-variance been previously reported in the literature?

A: Not to our knowledge, and we have asked many experts, but please bring any references to our attention. If anyone has made the exact same prediction using a model then we will announce them the winner.

Q: Does q-variance have implications for quantitative finance?

A: Yes, classical finance assumes a diffusive model for price change, but q-variance is a marker of a different kind of price dynamics that is shaped by transactions. Standard formulas such as Black-Scholes or the formula used to calculate VIX will therefore not work as expected.

Q: How does q-variance vary over different time periods, or from stock to stock?

A: In theory the curve should be time-invariant, though in practice there is a small degree of variation, see [Figure 2](Figure_2.png). The results for individual stocks are of course noisier and have a different minimum volatility as shown in [Figure 3](Figure_3.png), but taking the average variance over a number of stocks smooths out this noise. You can experiment further using the [Qvar Shiny app](https://david-systemsforecasting.shinyapps.io/qvar/).

Q: Is q-variance related to the implied volatility smile?

A: Yes. It is not the same thing because q-variance applies to realized volatility – but if you want to model implied volatility, a first step obviously is to understand realized volatility.

Q: Is q-variance related to the price-change distribution over a period?

A: Yes, it implies that price-change follows the q-distribution which is a particular time-invariant, Poisson-weighted sum of Gaussians (see further reading below). [Figure 4](Figure_4.png) compares the q-distribution with the average distribution over the S&P 500 stocks, where the distribution of each stock has been normalized by its standard deviation for comparability. The time-invariance is illustrated in [Figure 5](Figure_5.png) for different periods T. If your model matches q-variance and is time-invariant then it should produce the q-distribution.

Q: Can I use AI for the challenge?

A: Yes, AI-assisted entries are encouraged. We used Grok to help design and code the challenge. Its [entry](submissions/grok_rough_vol) is a modified rough volatility model which achieves an R² of 0.986, however it does use four parameters and also is not time-invariant. The aim is to find a process which can achieve better results with fewer parameters.

## Further Reading

Orrell D (2025) A Quantum Jump Model of Option Pricing. The Journal of Derivatives 33(2).

Orrell D (2025) Quantum impact and the supply-demand curve. Philosophical Transactions of the Royal Society A 383(20240562).

Orrell D (2026) The Quantum Stock Market. MIT Press (in press).

Wilmott P, Orrell D (2025) [Q-Variance: or, a Duet Concerning the Two Chief World Systems](Q-Variance_Wilmott_July2025.pdf). Wilmott 2025(138).
 
Visit the [Qvar Shiny app](https://david-systemsforecasting.shinyapps.io/qvar/) to do more simulations.
