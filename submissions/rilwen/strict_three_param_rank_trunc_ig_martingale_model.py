#!/usr/bin/env python3
"""
strict_three_param_rank_trunc_ig_martingale_model.py

Strict three-parameter martingale-corrected rank-truncated inverse-gamma q-variance generator.

Motivation:
    The raw inverse-gamma quantile model can hit the q-variance R2 target, but
    at 5M length it can generate price overflows. This version keeps the
    inverse-gamma mechanism but replaces the raw Gaussian CDF tail by a
    deterministic finite-sample rank regularisation.

Exposed model parameters:
    alpha
    beta_mult
    eta

Fixed architecture:
    omega_t = fixed multiscale Gaussian volatility field over tau in [1,252]
    rank_u_t = empirical rank of omega_t in (0,1)
    eps_N = N^{-1/2}
    u_t = eps_N + (1 - 2 eps_N) rank_u_t
    V_t = InvGamma^{-1}(u_t; alpha, beta_mult * SIGMA0^2)

    A_t = exp(-eta eps_t - eta^2/2)
    r_t = sqrt(V_t A_t / 252) eps_t

The finite-sample rank regularisation is not an additional fitted parameter:
it is a deterministic function of the requested sample length N. Whether this
is accepted as a strict-rule architecture is still a competition-rule question,
but it is cleaner than adding an adjustable cap.

Outputs:
    <out_prefix>_summary.csv
    <out_prefix>_perT.csv
    optional best finite price CSV
"""

import argparse
import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import lfilter

SIGMA0 = 0.2586
Z0 = 0.0214
HORIZONS = 5 * (np.arange(26) + 1)

# Fixed architecture constants.
KERNEL_K = 32
TAU_MIN = 1.0
TAU_MAX = 252.0
KERNEL_POWER = 0.5

ZMAX = 0.6
DELZ = 0.05


def qvar(z):
    return SIGMA0**2 + 0.5 * (z - Z0)**2


def simulate_ar1_factor(n, phi, rng):
    eps = rng.standard_normal(n)
    scale = math.sqrt(max(1.0 - phi * phi, 0.0))
    x = lfilter([scale], [1.0, -phi], eps)
    x -= x.mean()
    s = x.std()
    if s > 0:
        x /= s
    return x


def make_omega(n, rng):
    taus = np.exp(np.linspace(np.log(TAU_MIN), np.log(TAU_MAX), KERNEL_K))
    weights = taus ** KERNEL_POWER
    weights = weights / math.sqrt(float(np.sum(weights**2)))

    omega = np.zeros(n)
    for tau, w in zip(taus, weights):
        phi = math.exp(-1.0 / tau)
        omega += w * simulate_ar1_factor(n, phi, rng)

    omega -= omega.mean()
    s = omega.std()
    if s > 0:
        omega /= s

    return omega


def rank_regularised_uniform(x):
    """
    Convert x to empirical ranks in (0,1), then shrink away from 0 and 1.

    eps_N = N^{-1/2} is a deterministic finite-sample tail regularisation.
    It prevents the model from sampling inverse-gamma probabilities more extreme
    than the effective resolution of a finite N path.
    """
    n = len(x)

    # argsort twice gives ranks 0..n-1.
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)

    rank_u = (ranks + 0.5) / n

    eps_n = n ** (-0.5)
    u = eps_n + (1.0 - 2.0 * eps_n) * rank_u
    return np.clip(u, 1e-12, 1.0 - 1e-12)


def make_vol_budget(n, rng, alpha, beta_mult):
    beta = beta_mult * SIGMA0**2
    omega = make_omega(n, rng)
    u = rank_regularised_uniform(omega)
    return stats.invgamma.ppf(u, a=alpha, scale=beta)


def generate_returns(n, seed, alpha, beta_mult, eta):
    rng = np.random.default_rng(seed)

    V = make_vol_budget(n, rng, alpha, beta_mult)
    eps = rng.standard_normal(n)

    # Martingale-corrected same-shock skew/leverage transform.
    #
    # The earlier transform sqrt(A)*eps with
    #     A = exp(-eta*eps - eta^2/2)
    # has the right skew effect but a small non-zero mean. Over 5M steps that
    # mean accumulates and can make the price file overflow even when the
    # q-variance shape is good.
    #
    # Define Y = eps * exp(-eta*eps/2 - eta^2/4).
    # For eps ~ N(0,1):
    #     E[Y]   = -eta/2 * exp(-eta^2/8)
    #     E[Y^2] = 1 + eta^2
    # We subtract E[Y] and divide by the exact standard deviation so the
    # conditional return has mean zero and variance V/252. No extra parameter.
    raw = eps * np.exp(-0.5 * eta * eps - 0.25 * eta * eta)
    mean_raw = -0.5 * eta * np.exp(-0.125 * eta * eta)
    var_raw = 1.0 + eta * eta - mean_raw * mean_raw
    shock = (raw - mean_raw) / np.sqrt(var_raw)

    r = np.sqrt(V / 252.0) * shock
    return r


def make_prices(returns, start_price=100.0):
    logp = np.empty(len(returns) + 1)
    logp[0] = 0.0
    logp[1:] = np.cumsum(returns)

    mid = 0.5 * (np.min(logp) + np.max(logp))
    shifted = logp - mid + math.log(start_price)

    if np.max(shifted) > 709 or np.min(shifted) < -740:
        raise ValueError(
            f"Price path cannot be represented safely: shifted log range "
            f"[{np.min(shifted):.2f}, {np.max(shifted):.2f}]"
        )

    prices = np.exp(shifted)

    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("Non-finite or non-positive price produced.")

    return prices


def price_safety_stats(returns):
    logp = np.empty(len(returns) + 1)
    logp[0] = 0.0
    logp[1:] = np.cumsum(returns)

    log_min = float(np.min(logp))
    log_max = float(np.max(logp))
    log_range = log_max - log_min

    return {
        "log_min": log_min,
        "log_max": log_max,
        "log_range": float(log_range),
        "max_abs_return": float(np.max(np.abs(returns))),
        "q999_abs_return": float(np.quantile(np.abs(returns), 0.999)),
        "finite_price_possible": bool(log_range < 1300.0),
    }


def windows_from_returns(returns):
    rows = []
    scale = math.sqrt(252.0)

    for T in HORIZONS:
        nwin = len(returns) // T
        arr = returns[:nwin * T].reshape(nwin, T)

        x = arr.sum(axis=1)
        sigma = arr.std(axis=1, ddof=0) * scale
        z_raw = x / math.sqrt(T / 252.0)
        z = z_raw - z_raw.mean()
        var = sigma * sigma

        rows.append(pd.DataFrame({"T": int(T), "z": z, "var": var}))

    return pd.concat(rows, ignore_index=True)


def binned_curve(df, zmax=ZMAX, delz=DELZ, min_count=1):
    bins = np.linspace(-zmax, zmax, int(round(2 * zmax / delz)) + 1)

    d = df[np.isfinite(df["z"]) & np.isfinite(df["var"]) & (df["var"] > 0)].copy()
    d = d[(d["z"] >= -zmax) & (d["z"] <= zmax)]

    if len(d) == 0:
        return pd.DataFrame(columns=["z_mid", "var", "n"])

    d["z_bin"] = pd.cut(d["z"], bins=bins, include_lowest=True)

    b = (
        d.groupby("z_bin", observed=False)
        .agg(z_mid=("z", "mean"), var=("var", "mean"), n=("var", "size"))
        .dropna()
        .reset_index(drop=True)
    )

    return b[b["n"] >= min_count].copy()


def fixed_r2(b):
    z = b["z_mid"].to_numpy()
    y = b["var"].to_numpy()
    yhat = qvar(z)

    den = np.sum((y - y.mean())**2)
    if den <= 0:
        return np.nan

    return 1.0 - np.sum((y - yhat)**2) / den


def free_fit(b):
    z = b["z_mid"].to_numpy()
    y = b["var"].to_numpy()

    X = np.column_stack([np.ones_like(z), z, z * z])
    A, L, B = np.linalg.lstsq(X, y, rcond=None)[0]

    yhat = X @ np.array([A, L, B])
    den = np.sum((y - y.mean())**2)
    free_r2 = 1.0 - np.sum((y - yhat)**2) / den if den > 0 else np.nan

    c = -L / (2.0 * B) if B != 0 else np.nan
    a = A - B * c * c if np.isfinite(c) else np.nan

    return a, B, c, free_r2


def diagnostics(returns):
    df = windows_from_returns(returns)
    pooled = binned_curve(df)

    a, B, c, free_r2 = free_fit(pooled)

    summary = {
        "windows_total": len(df),
        "pooled_fixed_R2": fixed_r2(pooled),
        "pooled_free_R2": free_r2,
        "pooled_a": a,
        "pooled_B": B,
        "pooled_c": c,
    }

    rows = []
    for T in HORIZONS:
        b = binned_curve(df[df["T"] == T])
        aT, BT, cT, frT = free_fit(b)
        rows.append({
            "T": int(T),
            "fixed_R2": fixed_r2(b),
            "free_a": aT,
            "free_B": BT,
            "free_c": cT,
            "free_R2": frT,
        })

    return summary, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alphas", default="")
    parser.add_argument("--betas", default="")
    parser.add_argument("--etas", default="")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta-mult", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--out-prefix", default="strict3_ranktrunc_ig_martingale")
    parser.add_argument("--out-price", default="")
    args = parser.parse_args()

    if args.alpha is not None:
        alphas = [args.alpha]
    else:
        alphas = [float(x.strip()) for x in (args.alphas or "0.8,1.0,1.15,1.25,1.35,1.50").split(",") if x.strip()]

    if args.beta_mult is not None:
        betas = [args.beta_mult]
    else:
        betas = [float(x.strip()) for x in (args.betas or "0.5,0.65,0.75,0.88,1.0,1.15,1.3").split(",") if x.strip()]

    if args.eta is not None:
        etas = [args.eta]
    else:
        etas = [float(x.strip()) for x in (args.etas or "0,0.02,0.03,0.04,0.045,0.05,0.055,0.06,0.08,0.10").split(",") if x.strip()]

    summaries = []
    all_perT = []

    best = None
    best_returns = None

    for alpha in alphas:
        for beta in betas:
            for eta in etas:
                run_seed = args.seed + int(round(alpha * 1000)) + int(round(beta * 10000)) + int(round(eta * 100000))
                print(f"\nRunning alpha={alpha:g}, beta_mult={beta:g}, eta={eta:g}, n={args.n}")

                returns = generate_returns(args.n, run_seed, alpha, beta, eta)
                safety = price_safety_stats(returns)
                summ, perT = diagnostics(returns)

                row = {
                    "alpha": alpha,
                    "beta_mult": beta,
                    "eta": eta,
                    **summ,
                    **safety,
                }
                summaries.append(row)

                perT["alpha"] = alpha
                perT["beta_mult"] = beta
                perT["eta"] = eta
                all_perT.append(perT)

                print(pd.DataFrame([row]).to_string(index=False))

                if safety["finite_price_possible"]:
                    if best is None or row["pooled_fixed_R2"] > best["pooled_fixed_R2"]:
                        best = row
                        best_returns = returns.copy()

    summary = pd.DataFrame(summaries).sort_values(["finite_price_possible", "pooled_fixed_R2"], ascending=[False, False])
    perT = pd.concat(all_perT, ignore_index=True)

    summary_path = f"{args.out_prefix}_summary.csv"
    perT_path = f"{args.out_prefix}_perT.csv"

    summary.to_csv(summary_path, index=False)
    perT.to_csv(perT_path, index=False)

    print("\nSaved:")
    print(summary_path)
    print(perT_path)

    if best is not None:
        print("\nBest finite candidate:")
        print(pd.DataFrame([best]).to_string(index=False))

        if args.out_price:
            prices = make_prices(best_returns)
            pd.DataFrame({"Price": prices}).to_csv(args.out_price, index=False)
            print(f"Saved best finite price path to {args.out_price}")
    else:
        print("\nNo finite candidate found.")


if __name__ == "__main__":
    main()
