# The Q-Variance Challenge

<img src="Figure_1.png" width="800">

Can any continuous-time model, using no more than three free parameters, reproduce what may be the most **clear-cut empirical property of variance**, namely the parabolic relationship known as **q-variance**?

This states that, for a sufficiently large data set of stock prices, variance over a period $T$ is well-approximated by the equation

$\sigma^2(z) = \sigma_0^2 + \frac{(z-z_0)^2}{2}$

where $z = x/\sqrt{T}$, and $x$ is the log price change over the period, adjusted for drift (the parameter $z_0$ accounts for small asymmetries). The figure above illustrates q-variance for stocks from the S&P 500, and periods $T$ of 1-26 weeks. Blue points are variance vs $z$ for individual periods, blue line is average variance as a function of $z$, red line is the q-variance curve. 

Q-variance affects everything from option pricing to how we measure and talk about volatility. Read the [Q-Variance WILMOTT article](Q-Variance_Wilmott_July2025.pdf) for more details and examples. See the competition announcement (5-Dec-2025) in the WILMOTT forum [here](https://forum.wilmott.com/viewtopic.php?p=889508&sid=0eb1fdd23cee0e6824de7353248d2e22#p889503). For an update on submissions as of end-2025 see [here](subsummary.md).

To take part in the challenge, a suggested first step is to replicate the above figure using the code and market data supplied. Then repeat using simulated data from your model, and score it as described below.

**Prize:** One-year subscription to WILMOTT magazine and publication of the technique.

**Closing Date:** None.

For questions on the competition, email admin@wilmott.com.

## Repository Contents

The repository contains:
- Parquet file in three parts containing benchmark price data 1950-2025 for 401 stocks from the S&P 500 (stocks with less than 25 percent of dates excluded)
- Full dataset generator `data_loader.py` to show how the data was generated
- Baseline model fit `baseline/baseline_fit.py`
- Figures showing q-variance and R² value for the actual data
- Dataset generator `code/data_loader_csv.py` to load a CSV file of model price data and generate a parquet file
- Scoring engine `code/score_submission.py` for your model
- Jupyter notebook `notebooks/qvariance_single.ipynb` showing how to compute q-variance for a single asset

Dataset columns are ticker (str), date (date), T (int), sigma (float, annualized vol), z (float, scaled log return). Due to file size limitations, the parquet file is divided into three parts. Combine them with the command:
```python
df = pd.concat([pd.read_parquet("dataset_part1.parquet"),pd.read_parquet("dataset_part2.parquet"),pd.read_parquet("dataset_part3.parquet")])
```

Python dependencies: pip install yfinance pandas numpy scipy matplotlib pyarrow

## Scoring the Challenge

The challenge scores submissions on **one global R²** over the **entire dataset**. Since the q-variance parabola with $\sigma_0=0.259$ and $z_0 = 0.021$ gives a near-perfect fit (R² = 0.999) this curve can be used as a proxy for the real data. In other words, the aim is to fit the two-parameter parabola, using **up to three parameters** – must be easy, right?

To get started, a good first step is to replicate the q-variance curve using `baseline/baseline_fit.py` with the supplied `dataset.parquet` file. You can also check out `notebooks/qvariance_single.ipynb` which shows how q-variance is computed for a single asset, in this case the S&P 500.

Next, simulate a long series of daily prices using your model, and save as a CSV file with a column named 'Price'. Use `data_loader_csv.py` to compute the variances $\sigma^2(z)$ for each window and output your own `dataset.parquet` file. The benchmark file has around 3 million rows, so you want a long simulation.

Finally, use `score_submission.py` to read your `dataset.parquet` (must match format: ticker, date, T, z, sigma). This will bin the values of $z$ in the range from -0.6 to 0.6 as in the figure, and compute the average variance per bin. It also computes the R² of your binned averages to the q-variance curve $\sigma^2(z) = \sigma_0^2 + (z-z_0)^2/2$.

The threshold for the challenge is R² ≥ 0.995 with no more than three free parameters. A free parameter includes parameters in the model that, when modified within reasonable bounds, affect the score. This includes tuning parameters such as base volatility or drift, but also parameters which are specifically set within the model to achieve q-variance (and note that if the model is unstable even apparently innocuous settings can influence the results). The price-change distribution in $z$ should also be time-invariant, so the model should be independent of period length $T$. If your model doesn't tick all the boxes, please enter it anyway because it may qualify for an honourable mention.

To make your entry official:

1. Fork this repository
2. Place your model output in `submissions/your_team_name/` as:
   - `dataset.parquet` (must have columns: ticker, date, T, z, sigma)
   - CSV file of daily prices (must have column: Price)
3. Add a `README.md` in your folder with:
   - Team name
   - Short model description
   - Contact (optional)
4. Open a Pull Request titled: "Submission: [Your Team Name]"

## Frequently Asked Questions

Q: Is q-variance a well-known "stylized fact"?

A: No, a stylized fact is a general observation about market data, but q-variance is a **falsifiable prediction** because the multiplicative constant on the quadratic term is not a fit, it is set by theory at 0.5. The same formula applies for all period lengths T. As far as we are aware this is the most clear-cut and easily tested example of a model prediction in finance.

Q: Is it only noticeable over very long time series, or by averaging the results from hundreds of different stocks?

A: No, you can see q-variance over normal time scales such as 20 years of data. It holds not just for stocks, but even for things like Bitcoin or bond yields (see the [article](Q-Variance_Wilmott_July2025.pdf)). If your model only seems to show q-variance over much longer simulations then it will be sensitive to small changes (e.g. to the exact simulation time) and it also won't be realistic.

Q: Is q-variance about implied volatility?

A: No, it is about asset price volatility. Q-variance does not involve option prices or implied volatility. There is a direct connection between q-variance and the implied volatility smile, but that is not the subject of this competition.

Q: Has q-variance been previously reported in the literature?

A: Not to our knowledge, and we have asked many experts, but please bring any references to our attention. If anyone has made the exact same prediction using a model then we will announce them the winner.

Q: Is q-variance a large effect?

A: Yes, the minimum variance is about half the total variance so this is a large effect. If you are modelling variance then you do need to take q-variance into account.

Q: Does q-variance have implications for quantitative finance?

A: Yes, classical finance assumes a diffusive model for price change, but q-variance is a marker of a different kind of price dynamics that is shaped by transactions. Standard formulas such as Black-Scholes or the formula used to calculate VIX will therefore not work as expected.

Q: How does q-variance vary over different time periods, or from stock to stock?

A: In theory the curve should be time-invariant, though in practice there is a small degree of variation with period length, see [Figure 2](Figure_2.png). The results for individual stocks are of course noisier and have a different minimum volatility as shown in [Figure 3](Figure_3.png), but taking the average variance over a number of stocks smooths out this noise. The curve is based on a first-order approximation to dynamics, and can hold less well for example when volatility is very low. You can experiment further using the [Qvar Shiny app](https://david-systemsforecasting.shinyapps.io/qvar/).

Q: Is q-variance related to the price-change distribution over a period?

A: Yes, it implies that price-change follows the q-distribution which is a particular time-invariant, Poisson-weighted sum of Gaussians (see further reading below). [Figure 4](Figure_4.png) compares the q-distribution with the average distribution over the S&P 500 stocks. The time-invariance is illustrated in [Figure 5](Figure_5.png) for different periods $T$. If your model matches q-variance and is time-invariant then it should produce the q-distribution.

Q: How long a time series do we need?

A: To reproduce Figure 1 you will need around 5e6 time points. That works out to about 20K years of data. However it isn't very realistic if q-variance is only visible over extremely long time periods, because with stocks you can see it with less than 20 years of data. To test your model, divide the data into 500 segments, each in a column labelled "V1", "V2", etc., create your parquet file, and run `score_submission.py`. This will produce a plot like [Figure 3](Figure_3.png), where now the separate columns are treated as representing individual stocks.

Q: Why should I enter this competition?

A: For fun, the awesome prizes, an intellectual challenge, kudos, to defend the honour of classical finance ... but also because, if your existing model of volatility doesn't do q-variance, then it doesn't really model volatility.

Q: Can I use AI for the challenge?

A: Sure, in fact we used Grok to help design and code the challenge. Its [entry](submissions/grok_rough_vol) is a modified rough volatility model which achieves an R² of 0.986, however it needs four parameters and also is not time-invariant. The aim is to find a process which can achieve better results with fewer parameters.

Q: How is the competition going so far?

A: Some great tries but no clear winner, see the summary [here](subsummary.md).

Q: Okay, I'll bite. What is the quantum explanation?

A: Price change is like pushing on a spring. The linear restoring force gives you the square-root law of price impact. Integrating the force gives you the $z^2/2$ term in q-variance. But you need to use a probabilistic framework which accounts for dynamics. See sources below.

Q: Sounds like quantum woo to me.

A: ?

## Further Reading

Wilmott P, Orrell D (2025) [Q-Variance: or, a Duet Concerning the Two Chief World Systems](Q-Variance_Wilmott_July2025.pdf). Wilmott 2025(138).

Orrell D (2022) Quantum Economics and Finance: An Applied Mathematics Introduction, third edition. New York: Panda Ohana. 

Orrell D (2025) A Quantum Jump Model of Option Pricing. The Journal of Derivatives 33(2): 9-27.

Orrell D (2025) Quantum impact and the supply-demand curve. Philosophical Transactions of the Royal Society A 383(20240562).

Orrell D (2026) The Quantum Stock Market. MIT Press (in press).

Visit the [Qvar Shiny app](https://david-systemsforecasting.shinyapps.io/qvar/) to do more simulations.
