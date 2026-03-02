"""
End-to-end submission generator for the Q-Variance challenge.

Runs the full pipeline:
    1. Simulate 5M trading days of prices
    2. Save variance_timeseries.csv
    3. Process through windowing/scoring pipeline -> dataset.parquet
    4. Save first 100K days as prices_100k.csv

Usage:
    python generate_submission.py
    python generate_submission.py --days 5000000 --seed 42
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from model import QuantumPrecisionProcess


HORIZONS = 5 * (np.arange(26) + 1)  # T = 5, 10, ..., 130


def simulate_prices(sigma0, kappa, rho, alpha, n_max, n_days, seed, burn_in):
    """Simulate price series using the Quantized Precision model."""
    model = QuantumPrecisionProcess(
        sigma0=sigma0, kappa=kappa, rho=rho,
        alpha=alpha, N_max=n_max,
    )
    print(f"Model: {model}")
    stats = model.state_statistics()
    print(f"  alpha={stats['alpha']:.4f}  E[V]={stats['E[V]']:.4f}")

    t0 = time.time()
    df = model.simulate(n_days=n_days, seed=seed, burn_in=burn_in)
    elapsed = time.time() - t0
    print(f"Simulated {n_days:,} days in {elapsed:.1f}s")
    return df


def build_dataset(prices):
    """Process price series into windowed dataset (replicates data_loader_csv.py)."""
    ret = np.diff(np.log(prices))
    scale = np.sqrt(252)
    rows = []

    for T in HORIZONS:
        i = 0
        while i + T <= len(ret):
            window = ret[i:i + T]
            x = window.sum()
            sigma = np.std(window, ddof=0) * scale
            z_raw = x / np.sqrt(T / 252.0)

            if np.isfinite(sigma) and sigma > 0 and np.isfinite(z_raw):
                rows.append({
                    "ticker": "Model",
                    "date": i + T - 1,
                    "T": T,
                    "z_raw": float(z_raw),
                    "sigma": float(sigma),
                })
            i += T

    df = pd.DataFrame(rows)
    df = df[np.isfinite(df["z_raw"]) & np.isfinite(df["sigma"]) & (df["sigma"] > 0)]

    # De-mean z within each T group
    df["z"] = df.groupby(["ticker", "T"])["z_raw"].transform(lambda g: g - g.mean())
    df = df.drop(columns="z_raw")
    df = df.dropna().reset_index(drop=True)

    return df


def score_dataset(df, target_s0=0.2586, target_zoff=0.0214):
    """Score the dataset against the competition target parabola."""
    df = df.copy()
    df["var"] = df["sigma"] ** 2

    zmax = 0.6
    delz = 0.025 * 2
    nbins = int(2 * zmax / delz + 1)
    bins = np.linspace(-zmax, zmax, nbins)

    binned = (
        df.assign(z_bin=pd.cut(df.z, bins=bins, include_lowest=True))
        .groupby("z_bin", observed=False)
        .agg(z_mid=("z", "mean"), var=("var", "mean"))
        .dropna()
    )

    def qvar(z, s0, zoff):
        return s0**2 + (z - zoff)**2 / 2

    fitted = qvar(binned.z_mid.values, target_s0, target_zoff)
    var_vals = binned["var"].values
    ss_res = np.sum((var_vals - fitted) ** 2)
    ss_tot = np.sum((var_vals - var_vals.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    return r2


def main():
    parser = argparse.ArgumentParser(description="Generate Q-Variance submission")
    parser.add_argument("--days", type=int, default=5_000_000)
    parser.add_argument("--sigma0", type=float, default=0.2691)
    parser.add_argument("--kappa", type=float, default=1.55)
    parser.add_argument("--rho", type=float, default=0.41)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--nmax", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--burn-in", type=int, default=1000)
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Simulate
    print("=" * 60)
    print("STEP 1: Simulate prices")
    print("=" * 60)
    sim_df = simulate_prices(
        args.sigma0, args.kappa, args.rho, args.alpha,
        args.nmax, args.days, args.seed, args.burn_in,
    )
    prices = sim_df["Price"].values

    # 2. Save variance_timeseries.csv
    csv_path = os.path.join(out_dir, "variance_timeseries.csv")
    sim_df[["Price"]].to_csv(csv_path, index=False)
    print(f"Saved {csv_path} ({len(prices):,} rows)")

    # 3. Save prices_100k.csv (first 100K days)
    n_sample = min(100_000, len(prices))
    prices_100k_path = os.path.join(out_dir, "prices_100k.csv")
    sim_df[["Price"]].iloc[:n_sample].to_csv(prices_100k_path, index=False)
    print(f"Saved {prices_100k_path} ({n_sample:,} rows)")

    # 4. Build dataset
    print("\n" + "=" * 60)
    print("STEP 2: Build windowed dataset")
    print("=" * 60)
    dataset = build_dataset(prices)
    parquet_path = os.path.join(out_dir, "dataset.parquet")
    dataset.to_parquet(parquet_path, compression=None)
    print(f"Saved {parquet_path} ({len(dataset):,} windows)")

    # 5. Score
    print("\n" + "=" * 60)
    print("STEP 3: Score against target")
    print("=" * 60)
    r2 = score_dataset(dataset)
    print(f"R^2 = {r2:.6f}")
    if r2 >= 0.995:
        print("PASS: R^2 >= 0.995")
    else:
        print(f"WARNING: R^2 = {r2:.4f} < 0.995")

    # Summary
    ann_vol = np.std(np.diff(np.log(prices))) * np.sqrt(252)
    print(f"\nSummary:")
    print(f"  Days simulated: {len(prices):,}")
    print(f"  Annualized vol: {ann_vol:.4f}")
    print(f"  Windows: {len(dataset):,}")
    print(f"  R^2: {r2:.6f}")
    print(f"\nFiles generated:")
    print(f"  {csv_path}")
    print(f"  {prices_100k_path}")
    print(f"  {parquet_path}")


if __name__ == "__main__":
    main()
