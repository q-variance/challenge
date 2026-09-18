"""Window statistics and headless scoring, matching q-variance/challenge code.

- window_stats reproduces code/data_loader_csv.py (np.std ddof=0, z = x / sqrt(T/252),
  z de-meaned per T).
- score reproduces the numerical outputs of code/score_submission.py: the R^2 of the
  binned variance against the fixed parabola (sigma0=0.2586, z0=0.0214) on |z|<=0.6,
  and the per-T regression slope b(T) of var on z^2 with its MAE vs 0.5.
"""
import numpy as np
import pandas as pd

DAYS = 252
HORIZONS = 5 * (np.arange(26) + 1)
S0, Z0 = 0.2586, 0.0214


def qvar(z, s0=S0, z0=Z0):
    return s0 ** 2 + (z - z0) ** 2 / 2


def window_stats(r, T):
    """r: (n_windows, T) daily log returns -> DataFrame of T, z_raw, sigma."""
    x = r.sum(axis=1)
    sigma = r.std(axis=1, ddof=0) * np.sqrt(DAYS)
    z = x / np.sqrt(T / DAYS)
    return pd.DataFrame({"T": T, "z_raw": z, "sigma": sigma})


def series_windows(ret, horizons=HORIZONS, ticker="Model"):
    """Non-overlapping windows of a single daily log-return series, as data_loader_csv.py."""
    out = []
    for T in horizons:
        n = len(ret) // T
        df = window_stats(ret[: n * T].reshape(n, T), T)
        df["date"] = np.arange(1, n + 1) * T - 1
        out.append(df)
    return finalise(pd.concat(out, ignore_index=True), ticker)


def finalise(df, ticker="Model"):
    df = df[np.isfinite(df.z_raw) & np.isfinite(df.sigma) & (df.sigma > 0)].copy()
    df["z"] = df.groupby("T")["z_raw"].transform(lambda g: g - g.mean())
    df["ticker"] = ticker
    if "date" not in df:
        df["date"] = df.groupby("T").cumcount()
    return df[["ticker", "date", "T", "z", "sigma"]].reset_index(drop=True)


def binned(df, zmax=0.6, delz=0.05):
    bins = np.linspace(-zmax, zmax, int(2 * zmax / delz + 1))
    v = df.sigma ** 2
    return (df.assign(var=v, z_bin=pd.cut(df.z, bins=bins, include_lowest=True))
              .groupby("z_bin", observed=False)
              .agg(z_mid=("z", "mean"), var=("var", "mean"), n=("z", "size"))
              .dropna())


def r2_parabola(b, s0=S0, z0=Z0):
    f = qvar(b.z_mid, s0, z0)
    return 1 - np.sum((b["var"] - f) ** 2) / np.sum((b["var"] - b["var"].mean()) ** 2)


def fit_parabola(b):
    """Free fit var = s0^2 + q (z - z0)^2 to binned data -> (s0, z0, q)."""
    from scipy.optimize import curve_fit
    f = lambda z, s0, z0, q: s0 ** 2 + q * (z - z0) ** 2
    p, _ = curve_fit(f, b.z_mid, b["var"], p0=[0.25, 0.0, 0.5])
    return p


def slopes(df):
    """Per-T OLS slope b(T) of var on z^2 over all points (as score_submission.py)."""
    rows = []
    for T, g in df.groupby("T"):
        b, a = np.polyfit(g.z ** 2, g.sigma ** 2, 1)
        rows.append((T, a, b))
    s = pd.DataFrame(rows, columns=["T", "a", "b"])
    return s, float(np.mean(np.abs(s.b - 0.5)))


def score(df, verbose=True):
    b = binned(df)
    r2 = r2_parabola(b)
    s0, z0, q = fit_parabola(b)
    s, mae = slopes(df)
    res = dict(n=len(df), r2=r2, fit_s0=s0, fit_z0=z0, fit_q=q, mae_b=mae,
               b5=s.b.iloc[0], b130=s.b.iloc[-1])
    if verbose:
        print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in res.items()))
    return res
