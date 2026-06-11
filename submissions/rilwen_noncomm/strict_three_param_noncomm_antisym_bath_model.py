#!/usr/bin/env python3
"""
strict_three_param_noncomm_antisym_bath_model.py

Three-parameter non-inverse-gamma open-system bath with commutator plus
orthogonalized symmetric order-flow correction.

This extends the best current branch:

    field = standardize(F0 + C_perp)

where F0 is the h2_tanh baseline bath and C_perp is the orthogonalized
commutator bath.  That branch was close to the target but appeared capped
around fixed R2 ~ 0.992.

The extra fixed architecture tested here is the symmetric order-flow term:

    S_t = E_{t-1} O_{t-2} + O_{t-1} E_{t-2}

This is the anticommutator-like partner to the commutator

    C_t = E_{t-1} O_{t-2} - O_{t-1} E_{t-2}

The symmetric term is Gram-Schmidt orthogonalized against the baseline and
commutator baths, so it cannot simply rescale the existing fit.

Exposed parameters:
    beta_mult
    memory
    eta

All other weights are fixed or eta-tied architecture choices.  Once a mode is
selected and hard-coded, the model is a clean three-parameter process.

Modes:
    unit_orth        control: F0 + C_perp
    sym_unit         F0 + C_perp + S_perp
    sym_tanh         F0 + C_perp + tanh(eta) S_perp
    sym_halfeta      F0 + C_perp + 0.5 eta S_perp
    sym_neg_tanh     F0 + C_perp - tanh(eta) S_perp
    sym_small        F0 + C_perp + sqrt(1-rho)^(1/2) S_perp

No inverse-gamma, no fitted fourth coefficient, no fitted second memory.
"""

from __future__ import annotations

import argparse
import math
from typing import List

import numpy as np
import pandas as pd

try:
    from scipy.signal import lfilter
except Exception as exc:
    raise SystemExit("Install dependencies with: pip install scipy pandas numpy") from exc


SIGMA0 = 0.2586
YEAR_DAYS = 252.0

VALID_MODES = {
    "unit_orth",
    "sym_unit",
    "sym_tanh",
    "sym_halfeta",
    "sym_neg_tanh",
    "sym_small",
}


def parse_grid(text: str, default: str) -> List[float]:
    src = text if text else default
    return [float(x.strip()) for x in src.split(",") if x.strip()]


def make_prices(returns: np.ndarray, p0: float = 100.0) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    logp = np.cumsum(r)
    lo = float(np.min(logp))
    hi = float(np.max(logp))
    mid = 0.5 * (lo + hi)
    return np.exp(math.log(p0) + logp - mid)


def price_safety_stats(returns: np.ndarray) -> dict:
    r = np.asarray(returns, dtype=float)
    logp = np.cumsum(r)
    finite = bool(np.all(np.isfinite(logp)))
    log_range = float(np.max(logp) - np.min(logp)) if finite else float("inf")
    return {
        "finite_price_possible": bool(finite and log_range < 1300.0),
        "log_range": log_range,
        "max_abs_return": float(np.nanmax(np.abs(r))),
        "q999_abs_return": float(np.nanquantile(np.abs(r), 0.999)),
        "q9999_abs_return": float(np.nanquantile(np.abs(r), 0.9999)),
    }


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - float(np.mean(x))
    s = float(np.std(x, ddof=0))
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("cannot standardize degenerate array")
    return x / s


def rho_from_memory(memory: float) -> float:
    rho = math.exp(-1.0 / float(memory))
    return min(max(rho, 0.0), 0.9999995)


def _ar1_from_drive(drive: np.ndarray, memory: float) -> np.ndarray:
    if memory <= 0.0:
        raise ValueError("memory must be positive")
    rho = rho_from_memory(memory)
    sigma = math.sqrt(max(0.0, 1.0 - rho * rho))
    y = lfilter([sigma], [1.0, -rho], np.asarray(drive, dtype=float))
    return _standardize(y)


def lag(x: np.ndarray, k: int) -> np.ndarray:
    y = np.zeros_like(x)
    if k <= 0:
        return x.copy()
    y[k:] = x[:-k]
    return y


def even_odd(eps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    E = (eps * eps - 1.0) / math.sqrt(2.0)
    O = eps
    return _standardize(E), _standardize(O)


def base_driver(eps: np.ndarray, eta: float) -> np.ndarray:
    E, O = even_odd(eps)
    D = E - math.tanh(float(eta)) * O
    return lag(_standardize(D), 1)


def order_drivers(eps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    E, O = even_odd(eps)
    a = lag(E, 1) * lag(O, 2)
    b = lag(O, 1) * lag(E, 2)
    C = _standardize(a - b)
    S = _standardize(a + b)
    return C, S


def gram_schmidt(x: np.ndarray, bases: list[np.ndarray]) -> np.ndarray:
    y = _standardize(x)
    for b in bases:
        bb = _standardize(b)
        coeff = float(np.mean(y * bb)) / max(float(np.mean(bb * bb)), 1e-12)
        y = y - coeff * bb
    return _standardize(y)


def sym_weight(memory: float, eta: float, mode: str) -> float:
    if mode == "unit_orth":
        return 0.0
    if mode == "sym_unit":
        return 1.0
    if mode == "sym_tanh":
        return math.tanh(float(eta))
    if mode == "sym_halfeta":
        return 0.5 * float(eta)
    if mode == "sym_neg_tanh":
        return -math.tanh(float(eta))
    if mode == "sym_small":
        rho = rho_from_memory(memory)
        gap = max(1.0 - rho, 1e-12)
        return gap ** 0.25
    raise ValueError(f"unknown mode {mode}")


def bath_field(eps: np.ndarray, memory: float, eta: float, mode: str) -> np.ndarray:
    F0 = _ar1_from_drive(base_driver(eps, eta), memory)
    C, S = order_drivers(eps)
    Fc_raw = _ar1_from_drive(C, memory)
    Fs_raw = _ar1_from_drive(S, memory)

    Fc = gram_schmidt(Fc_raw, [F0])
    Fs = gram_schmidt(Fs_raw, [F0, Fc])

    w = sym_weight(memory, eta, mode)
    return _standardize(F0 + Fc + w * Fs)


def activity_from_field(field: np.ndarray, eta: float) -> np.ndarray:
    logA = float(eta) * field
    # Numerical stabilization only.
    A = np.exp(logA - np.max(logA))
    A = A / float(np.mean(A))
    return A


def generate_returns(
    n: int,
    seed: int,
    beta_mult: float,
    memory: float,
    eta: float,
    mode: str = "unit_orth",
) -> np.ndarray:
    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode {mode}")
    if beta_mult <= 0.0:
        raise ValueError("beta_mult must be positive")
    if memory <= 0.0:
        raise ValueError("memory must be positive")
    if eta < 0.0:
        raise ValueError("eta must be nonnegative")

    rng = np.random.default_rng(int(seed))
    eps = rng.standard_normal(int(n))

    field = bath_field(eps, float(memory), float(eta), mode)
    A = activity_from_field(field, float(eta))

    r = np.sqrt(float(beta_mult) * SIGMA0 * SIGMA0 * A / YEAR_DAYS) * eps
    r = r - float(np.mean(r))
    return r


def empirical_aggs(_parts: str):
    return None, None


def score_returns(*_args, **_kwargs):
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=sorted(VALID_MODES), default="unit_orth")
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--betas", default="")
    ap.add_argument("--memories", default="")
    ap.add_argument("--etas", default="")
    ap.add_argument("--beta-mult", type=float, default=None)
    ap.add_argument("--memory", type=float, default=None)
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--out-prefix", default="strict3_noncomm_antisym_bath")
    ap.add_argument("--out-price", default="")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--print-every", type=int, default=20)
    args = ap.parse_args()

    betas = [args.beta_mult] if args.beta_mult is not None else parse_grid(args.betas, "2.00,2.03,2.05,2.07")
    memories = [args.memory] if args.memory is not None else parse_grid(args.memories, "85,95,100,105,115")
    etas = [args.eta] if args.eta is not None else parse_grid(args.etas, "1.09,1.10,1.105,1.11,1.12")

    rows = []
    best = None
    best_returns = None
    count = 0
    for beta in betas:
        for memory in memories:
            for eta in etas:
                count += 1
                try:
                    returns = generate_returns(args.n, args.seed, beta, memory, eta, mode=args.mode)
                    safety = price_safety_stats(returns)
                    row = {"mode": args.mode, "beta_mult": beta, "memory": memory, "eta": eta, **safety}
                except Exception as e:
                    if not args.quiet:
                        print(f"FAILED beta={beta:g} memory={memory:g} eta={eta:g}: {e}")
                    continue
                rows.append(row)
                show = (not args.quiet) or (args.print_every > 0 and count % args.print_every == 0)
                if show:
                    print(pd.DataFrame([row]).to_string(index=False))
                if safety.get("finite_price_possible", False):
                    if best is None or safety.get("log_range", float("inf")) < best.get("log_range", float("inf")):
                        best = row
                        best_returns = returns.copy()

    summary = pd.DataFrame(rows)
    if len(summary):
        summary = summary.sort_values(["finite_price_possible", "log_range"], ascending=[False, True])
    summary_path = f"{args.out_prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nSaved:")
    print(summary_path)
    if best is not None:
        print("\nRepresentative finite candidate:")
        print(pd.DataFrame([best]).to_string(index=False))
        if args.out_price:
            pd.DataFrame({"Price": make_prices(best_returns)}).to_csv(args.out_price, index=False)
            print(f"Saved price path to {args.out_price}")


if __name__ == "__main__":
    main()
