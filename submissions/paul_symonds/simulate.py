"""
CLI driver for the Quantized Precision model.

Generates price time series and saves to variance_timeseries.csv
in the format expected by the challenge's data_loader_csv.py.

Usage:
    python simulate.py --days 5000000 --seed 42
    python simulate.py --days 1000000 --sigma0 0.2695 --kappa 1.54 --rho 0.39
"""

import argparse
import time

import numpy as np

from model import QuantumPrecisionProcess


def main():
    parser = argparse.ArgumentParser(description="Simulate Quantized Precision model")
    parser.add_argument("--days", type=int, default=5_000_000,
                        help="Number of trading days (default: 5000000)")
    parser.add_argument("--sigma0", type=float, default=0.2691,
                        help="Base volatility (default: 0.2691)")
    parser.add_argument("--kappa", type=float, default=1.55,
                        help="Mean-reversion speed (default: 1.55)")
    parser.add_argument("--rho", type=float, default=0.41,
                        help="Leverage correlation (default: 0.41)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Shape parameter (default: 1.5)")
    parser.add_argument("--nmax", type=int, default=200,
                        help="Lattice levels (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--burn-in", type=int, default=1000,
                        help="Burn-in steps (default: 1000)")
    parser.add_argument("--output", type=str, default="variance_timeseries.csv",
                        help="Output CSV file path")
    args = parser.parse_args()

    model = QuantumPrecisionProcess(
        sigma0=args.sigma0,
        kappa=args.kappa,
        rho=args.rho,
        alpha=args.alpha,
        N_max=args.nmax,
    )

    print(f"Model: {model}")
    stats = model.state_statistics()
    print(f"  alpha={stats['alpha']:.4f}  dz={stats['dz']:.4f}")
    print(f"  E[V]={stats['E[V]']:.4f}  Var(V)={stats['Var(V)']:.6f}")
    print(f"  V range: [{stats['V_min']:.4f}, {stats['V_max']:.4f}]")
    print(f"Simulating {args.days:,} days...")

    t0 = time.time()
    df = model.simulate(
        n_days=args.days,
        dt=1 / 252,
        seed=args.seed,
        burn_in=args.burn_in,
    )
    elapsed = time.time() - t0
    print(f"Simulation complete: {len(df):,} rows in {elapsed:.1f}s")

    output_df = df[["Price"]].copy()
    output_df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

    prices = df["Price"].values
    log_rets = np.diff(np.log(prices))
    ann_vol = np.std(log_rets) * np.sqrt(252)
    print(f"\nSanity checks:")
    print(f"  Annualized vol: {ann_vol:.4f}")
    print(f"  Price range: [{prices.min():.4e}, {prices.max():.4e}]")


if __name__ == "__main__":
    main()
