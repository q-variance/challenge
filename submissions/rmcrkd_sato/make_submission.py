"""Build the challenge submission folder. Run from the repo root.

    .venv/bin/python scripts/make_submission.py [--days 5000000]

Writes submission/rmcrkd_sato/: dataset.parquet (anchored windows, primary model),
prices_100k.csv (single price series, Sato clock restarted every 130 days),
the model code (a copy of qfrh/) and results/*.json with all scores.
"""
import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STANDALONE = (HERE / "qfrh").is_dir()  # running from inside the submission folder
ROOT = HERE if STANDALONE else HERE.parent
sys.path.insert(0, str(ROOT))
from qfrh import analytic as an, qvar  # noqa: E402
from qfrh.experiments import anchored, block_series  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE if STANDALONE else ROOT / "submission" / "rmcrkd_sato"
P = json.load(open(HERE / "params.json" if STANDALONE else ROOT / "results" / "params.json"))
MODELS = {
    "meixner3": an.MeixnerSato(*P["mx3"]),  # primary entry
    "meixner2": an.MeixnerSato(*P["mx2"]),  # delta = 1/2 fixed by theory
    "frh": an.FRHSato(*P["frh"], nodes=16),  # rmcrkd/frh-fx, gamma(t) = c sqrt(t)
}
BLOCK = 130


def main(days):
    OUT.mkdir(parents=True, exist_ok=True)
    samplers = {k: an.IncrementSampler(m, 252) for k, m in MODELS.items()}
    sims = {k: (lambda s: lambda n, T, rng: s.sample(n, T, rng))(s) for k, s in samplers.items()}
    res = {}
    for k, m in MODELS.items():
        e = an.expected_score(m)
        res[k] = {"expected": {"r2": e["r2"], "mae_b": e["mae_b"]}}
        d = days if k == "meixner3" else days // 5
        df = anchored(sims[k], d, qvar.HORIZONS, seed=11, chunk=100_000)
        res[k]["anchored"] = qvar.score(df, verbose=False) | {"days": d}
        if k == "meixner3":
            df.to_parquet(OUT / "dataset.parquet")
        for B in (130, 252):
            r = block_series(sims[k], days // 5, B, seed=12)
            res[k][f"single_series_B{B}"] = qvar.score(qvar.series_windows(r), verbose=False)
        print(k, json.dumps(res[k], default=float, indent=1))

    r = block_series(sims["meixner3"], 100_000, BLOCK, seed=13)
    price = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    pd.DataFrame({"Price": price}).to_csv(OUT / "prices_100k.csv", index=False)

    json.dump({"params": P, "results": res}, open(OUT / "results.json", "w"), default=float, indent=1)
    if STANDALONE:
        return
    shutil.copytree(ROOT / "qfrh", OUT / "qfrh", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(__file__, OUT / "make_submission.py")
    shutil.copy(ROOT / "results" / "params.json", OUT / "params.json")
    for f in ("qvar_by_T.png", "b_of_T.png", "smiles.png"):
        shutil.copy(ROOT / "figures" / f, OUT / f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5_000_000)
    main(ap.parse_args().days)
