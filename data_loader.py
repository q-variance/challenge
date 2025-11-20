# data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

HORIZONS = 5*(np.arange(26)+1)   # does 1 to 26 weeks, can also do [5, 10, 20, 40, 80, 160]

TICKERS = ["^GSPC", "^DJI", "^FTSE", "AAPL", "MSFT", "AMZN", "JPM", "BTC-USD"]
# GSPC data from 1927-12-30, DJI from 1992-01-02, MSFT from 1986-03-13

Path("cache").mkdir(exist_ok=True)
all_data = []

print("Generating Q-Variance Challenge Dataset...")

for ticker in TICKERS:
    print(f"→ {ticker}", end="")
    price = yf.download(ticker, period="max", progress=False, auto_adjust=True)["Close"]
    ret = np.log(price).diff().dropna().values

    rows = []
    scale = np.sqrt(252)

    for T in HORIZONS:
        i = 0
        while i + T <= len(ret):
            window = ret[i:i+T]     # T points
            if len(window) < T * 0.8:
                break

            x = window.sum()   # total price change over the period
            sigma = np.std(window, ddof=0) * scale  # use np.std to get std over period, ddof=1 means divisor is N-1
            z_raw = x / np.sqrt(T / 252.0)

            # REJECT BAD WINDOWS
            if not (np.isfinite(sigma) and sigma > 0 and np.isfinite(z_raw)):
                i += T
                continue

            rows.append({          # append row of data for this period
                "ticker": ticker,
                "date": price.index[i + T - 1].date(),
                "T": T,
                "z_raw": float(z_raw),
                "sigma": float(sigma)
            })
            i += T

    if not rows:
        print(" [no data]")
        continue

    df = pd.DataFrame(rows)

    # CLEAN BEFORE DE-MEANING 
    df = df[np.isfinite(df['z_raw']) & np.isfinite(df['sigma']) & (df['sigma'] > 0)]

    # NOW de-mean safely, this step groups by ticker and T, and subtracts the group mean 
    df["z"] = df.groupby(["ticker", "T"])["z_raw"].transform(lambda g: g - g.mean())

    df = df.drop(columns="z_raw")
    df = df.dropna().reset_index(drop=True)  # Final clean

    df.to_parquet(Path("cache") / f"{ticker}.parquet")
    print(f" → {len(df)} clean windows")
    all_data.append(df)

full = pd.concat(all_data, ignore_index=True)
full.to_parquet("prize_dataset.parquet", compression=None)
print(f"\nSUCCESS! prize_dataset.parquet created with {len(full):,} clean rows")
#print(f"^GSPC T=20: {len(full[(full.ticker=='^GSPC') & (full.T==20)])} windows")
#print("z has NaNs:", full[(full.ticker=='^GSPC') & (full.T==20)]['z'].isna().sum())
