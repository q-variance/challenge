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

> ## Competition Update 11-01-2026
> **Read this before submitting!**
>
> The challenge has been running for over a month and we now have a good number of entries.
>
> So far, none of the entries match the q-variance curve to the target accuracy using no more than three parameters. Note that the aim of the challenge is to match the curve in the figure, so at a minimum you need a parameter which controls the minimum volatility, and another to produce a small offset. That only gives you one extra parameter to play with.
>
> We are therefore going to make some suggestions about approaches.
>
> **Inverse-gamma.** We have had several entries which work by drawing a stochastic volatility from an inverse-gamma distribution. Starting parameters are shape factor and rate for the distribution, plus the drift. Setting the shape factor to 3/2 reproduces q-variance perfectly in theory, the problem is trying to get a time series that matches it. This requires some kind of jump approach with another parameter (putting the total to four) that defines an inherent timescale related to the frequency of jumps (a problem since q-variance is time-invariant). The resulting model can take thousands of years to converge, and the log price change distribution is also too fat-tailed to be realistic.
>
> **GARCH(1,1).** You can get what seems to be a pretty good fit using this approach, but it requires four parameters just to get started: alpha, beta, xi, and a drift to match the offset. Matching q-variance again puts the parameters into an unstable regime with unbounded variance-of-variance, so the fit is sensitive to things like the simulation time, and you need extra parameters like a cap on volatility to stop so-called moment explosions.
>
> **Rough volatility.** This requires five parameters: a roughness index, leverage, vol-of-vol, initial variance, plus the drift to match the horizontal offset. It’s all going a bit [Von Neumann’s elephant](https://en.wikipedia.org/wiki/Von_Neumann%27s_elephant), and it still can’t match q-variance – which again is a basic empirical property of variance.
>
> So please don’t send more such models, unless you can think of some new spin which doesn’t just undercount the number of parameters.
>
> Because the goal may be unachievable – and also because we have noticed a tendency to game the challenge, for example by handpicking simulation times to achieve good results using unstable models – we are going to change the evaluation to more of a “beauty contest” approach where entries are rated according to their ability to converge to q-variance in the long-term (e.g. 1e6 days), but also work well for shorter simulations over reasonable simulation times like 10K days. In other words we are looking for realistic, transparent models that might actually be useful. Be sure to include some code (Python or preferably R, not Excel) so we can reproduce the results.
>
> In the meantime, we also hope this challenge leads to questions about the mindset in this field. Quant finance has usually been considered a “soft” science. This is why there isn’t one theory of volatility, there are by some counts over thirty. But q-variance is not a soft constraint, it is a hard one. Time to winnow the list down a bit.
>
> Finally, in case you think there is no model which matches q-variance, there obviously is because the property was predicted using a model. It’s just that, by design, it doesn’t work in continuous time. See the references below.

## Repository Contents

The repository contains:
- Parquet file in three parts containing benchmark price data 1950-2025 for 401 stocks from the S&P 500 (stocks with less than 25 percent of dates excluded)
- Full dataset generator `data_loader.py` to show how the data was generated
- Baseline model fit `baseline/baseline_fit.py`
- Figures showing q-variance and R² value for the actual data
- Dataset generator `code/data_loader_csv.py` to load a CSV file of model price data and generate a parquet file
- Scoring engine `code/score_submission.py` for your model
- Jupyter notebook `notebooks/qvariance_single.ipynb` showing how to compute q-variance for a single asset
- A folder `submissions' with current entries

Dataset columns are ticker (str), date (date), T (int), sigma (float, annualized vol), z (float, scaled log return). Due to file size limitations, the parquet file is divided into three parts. Combine them with the command:
```python
df = pd.concat([pd.read_parquet("dataset_part1.parquet"),pd.read_parquet("dataset_part2.parquet"),pd.read_parquet("dataset_part3.parquet")])
```

Python dependencies: pip install yfinance pandas numpy scipy matplotlib pyarrow

## Scoring the Challenge

The aim of the challenge is to replicate the empirical phenomenon of q-variance. Since the q-variance parabola with $\sigma_0=0.259$ and $z_0 = 0.021$ gives a near-perfect fit (R² = 0.999) this curve can be used as a proxy for the real data. In other words, the aim is to fit the two-parameter parabola, using **up to three parameters** – must be easy, right?

To get started, a good first step is to replicate the q-variance curve using `baseline/baseline_fit.py` with the supplied `dataset.parquet` file. You can also check out `notebooks/qvariance_single.ipynb` which shows how q-variance is computed for a single asset, in this case the S&P 500.

Next, simulate a long series of daily prices using your model, and save as a CSV file with a column named 'Price'. Use `data_loader_csv.py` to compute the variances $\sigma^2(z)$ for each window and output your own `dataset.parquet` file. To match the benchmark you will want a long simulation of around 5e6 days. Also save a shorter version with 100K rows that can be easily checked.

Finally, use `score_submission.py` to read your `dataset.parquet` (must match format: ticker, date, T, z, sigma). This will bin the values of $z$ in the range from -0.6 to 0.6 as in the figure, and compute the average variance per bin. It also computes the R² of your binned averages to the q-variance curve $\sigma^2(z) = \sigma_0^2 + (z-z_0)^2/2$.

The threshold for the challenge is R² ≥ 0.995 with no more than three free parameters. A free parameter includes parameters in the model that, when modified within reasonable bounds, affect the score. This includes tuning parameters such as base volatility or drift, but also parameters which are specifically set within the model to achieve q-variance (and note that if the model is unstable even apparently innocuous settings can influence the results). The aim is to fit the exact curve in Figure 1 with $z_0 = 0.021$, so you will need one parameter to achieve the small offset. Also, the simulation should be **robust to reasonable changes in the simulation length** (it is supposed to converge). The price-change distribution in $z$ should also be time-invariant, so the model should be independent of period length $T$. If your model doesn't tick all the boxes, please enter it anyway because it may qualify for an honourable mention.

To make your entry official:

1. Fork this repository
2. Place your model output in `submissions/your_team_name/` as:
   - `dataset.parquet` with all data (must have columns: ticker, date, T, z, sigma)
   - sample CSV file of daily prices for 100K days (must have column: Price)
   - code to produce a time series of daily prices and returns (Python or R)
3. Add a `README.md` in your folder with:
   - Team name
   - Short model description
   - Contact (optional)
4. Open a Pull Request titled: "Submission: [Your Team Name]"

**Submission tips**
- Read the [summary of previous submissions](subsummary.md) so you don't duplicate an existing approach.
- Check your model is robust to things like the number of simulation steps or sample lengths, otherwise these are counted as parameters. The model data should converge to the parabola, not just match it for a particular choice of simulation time.
- Something counts as a parameter if it is adjusted to fit the desired result, or if changing it within reasonable bounds affects the result. Please declare **all such parameters**, don't have an ad hoc number or adjustment somewhere in your code which affects the results but is not declared as a parameter. The test is not whether a number is explicitly optimized, or is at its "default value", it is whether it can be eliminated and the model still works.
- Parameters need to include a drift or offset, because the aim is to fit the specific parabola in Figure 1 which has a small offset of $z_0 = 0.021$. The minimum volatility should be $\sigma_0=0.259$.
- Standard techniques like stochastic volatility, rough volatility, or GARCH typically involve a minimum of four parameters when we include the drift (and often more to ensure a stable solution for this problem). That means you need to do something different.
- The model should be T-invariant, so changing the period T should not give a very different result, either in terms of the q-variance plot or the distribution.
- Make sure that you submit a sample time series and the code to replicate the model, not just a parquet file and a plot.
- Remember the parabola in Figure 1 matches real data, so to succeed your model will need to produce stable and realistic behaviour, otherwise changing internal numbers will affect the results and count as parameters. 


## Frequently Asked Questions

Q: Is q-variance a well-known "stylized fact"?

A: No, a stylized fact is a general observation about market data, but q-variance is a **falsifiable prediction** because the multiplicative constant on the quadratic term is not a fit, it is set by theory at 0.5. The same formula applies for all period lengths T. As far as we are aware this is the most clear-cut and easily tested example of a model prediction in finance. For some reason though conventional models don't show it.

Q: Is it only noticeable over very long time series, or by averaging the results from hundreds of different stocks?

A: No, you can see q-variance over normal time scales such as 20 years of data. It holds not just for stocks, but even for things like Bitcoin or bond yields (see the [article](Q-Variance_Wilmott_July2025.pdf)). If a model of it only works over much longer simulations then it will be sensitive to small changes (e.g. to the exact simulation time) and it also won't be realistic.

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

A: Yes, price-change in the quantum model follows the q-distribution which is a particular time-invariant, Poisson-weighted sum of Gaussians (see further reading below). [Figure 4](Figure_4.png) compares the q-distribution with the average distribution over the S&P 500 stocks. The time-invariance is illustrated in [Figure 5](Figure_5.png) for different periods $T$.

Q: How long a time series do we need?

A: To reproduce Figure 1 you will need around 5e6 days. That works out to about 20K years of data. However it isn't very realistic if q-variance is only visible over extremely long time periods, because with stocks you can see it with less than 20 years of data. To test your model, divide the data into 500 segments, each in a column labelled "V1", "V2", etc., create your parquet file, and run `score_submission.py`. This will produce a plot like [Figure 3](Figure_3.png), where now the separate columns are treated as representing individual stocks. In general, the result should be robust to reasonable changes in simulation length.

Q: Some parameters in my model were preset, do they still count towards the limit of three?

A: If changing them within reasonable bounds affects the result, then yes they count. Note that the aim is to fit the specific quadratic in the figure, so you will need a parameter to achieve the small horizontal offset. It's not enough to fit the case where there is no offset, otherwise we could remove the offset parameter $z_0 = 0.021$ from the q-variance curve also.

Q: Why should I enter this competition?

A: For fun, the awesome prizes, an intellectual challenge, kudos, to defend the honour of classical finance ... but also because, if your existing model of volatility doesn't do q-variance, then it doesn't really model volatility.

Q: Can I use AI for the challenge?

A: Sure, in fact we used Grok to help design and code the challenge. Its [entry](submissions/grok_rough_vol) is a modified rough volatility model which achieves an R² of 0.986, however it needs five parameters and also is not time-invariant. The aim is to find a process which can achieve better results with fewer parameters.

Q: How is the competition going so far?

A: Some great tries but no clear winner, see the summary [here](subsummary.md).

Q: Okay, I'll bite. What is the quantum explanation?

A: Price change is like stretching a spring. The linear restoring force gives you the square-root law of price impact. Integrating the force gives you the $z^2/2$ term in q-variance. The model only has one parameter (the offset is a small error term). But you need to use a probabilistic framework which accounts for dynamics. See sources below.

Q: Sounds like quantum woo to me.

A: ?

Q: Hang on. Why do we need a continuous time model anyway? Don't we just need the price change distribution as a function of time?

A: In theory you need it for perfect hedging, but in practice that is impossible due to things like the bid-ask spread. The situation may therefore resemble somthing like turbulent flow in physics, where we can model the statistical behaviour without reproducing individual trajectories. The aim of the challenge is to help answer that question.

## Further Reading

Wilmott P, Orrell D (2025) [Q-Variance: or, a Duet Concerning the Two Chief World Systems](Q-Variance_Wilmott_July2025.pdf). Wilmott 2025(138).

Orrell D (2022) Quantum Economics and Finance: An Applied Mathematics Introduction, third edition. New York: Panda Ohana. 

Orrell D (2025) A Quantum Jump Model of Option Pricing. The Journal of Derivatives 33(2): 9-27.

Orrell D (2025) Quantum impact and the supply-demand curve. Philosophical Transactions of the Royal Society A 383(20240562).

Orrell D (2026) The Quantum Stock Market. MIT Press (in press).

Visit the [Qvar Shiny app](https://david-systemsforecasting.shinyapps.io/qvar/) to do more simulations.
