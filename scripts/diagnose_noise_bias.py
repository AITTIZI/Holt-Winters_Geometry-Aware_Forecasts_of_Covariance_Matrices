from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import digamma  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import CONFIGS, MODEL_ORDER, RunSettings  # noqa: E402
from src.geometry import airm_distance  # noqa: E402
from src.metrics import rmse  # noqa: E402
from src.models import MODEL_CLASSES  # noqa: E402

SCENARIOS = ("cir", "jump", "rotation", "rotation_ou")

def theoretical_bias(nu: int, n: int) -> float:
    return float(sum(digamma((nu - i + 1) / 2) for i in range(1, n + 1)) + n * np.log(2 / nu))

def logdet(A: np.ndarray) -> np.ndarray:
    return np.linalg.slogdet(A)[1]


def _replication(task: tuple) -> list[dict]:
    scenario, rep, Sigma, obs, settings = task
    config = CONFIGS["noisy"]
    n = Sigma.shape[-1]
    theory = theoretical_bias(config.noise_dof, n)
    rows = []
    for name in MODEL_ORDER:
        model = MODEL_CLASSES[name](config.spec).fit(obs[: settings.n_train])
        result = model.filter(obs)
        row = {"scenario": scenario, "rep": rep, "model": name}
        for h in (1, 10):
            origins = np.arange(settings.n_train - 1, settings.n_steps - h)
            F, target = model.forecast(result, origins, [h])[:, 0], Sigma[origins + h]
            ok = np.linalg.eigvalsh(F)[:, 0] > 0
            row[f"logdet_bias_h{h}"] = float(np.mean(logdet(F[ok]) - logdet(target[ok]))) if ok.any() else np.nan
            if model.preserves_spd:
                row[f"airm_rmse_h{h}"] = rmse(airm_distance(F, target))
                row[f"airm_rmse_corrected_h{h}"] = rmse(airm_distance(F * np.exp(-theory / n), target))
        rows.append(row)
    return rows


def run(results: Path, workers: int | None = None) -> pd.DataFrame:

    meta_settings = RunSettings()
    nu = CONFIGS["noisy"].noise_dof
    tasks = []
    for scenario in SCENARIOS:
        path = results / "data" / f"{scenario}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        if f"obs_nu{nu}" not in data.files:
            continue
        tasks += [(scenario, int(rep), data["Sigma"][i], data[f"obs_nu{nu}"][i], meta_settings) for i, rep in enumerate(data["reps"])]
    if not tasks:
        raise SystemExit("no saved noisy paths found: run run_experiments.py (with the noisy configuration) first")
    with mp.get_context("spawn").Pool(workers or os.cpu_count()) as pool:
        rows = [r for chunk in pool.imap_unordered(_replication, tasks, chunksize=4) for r in chunk]
    df = pd.DataFrame(rows).sort_values(["scenario", "rep", "model"])
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    table = df.groupby(["scenario", "model"], sort=False).mean(numeric_only=True).drop(columns="rep").reset_index()
    table = table.sort_values(["scenario", "model"], key=lambda c: c.map(order) if c.name == "model" else c)
    table["theory_single_obs"] = theoretical_bias(nu, 3)
    out = results / "summary" / "noise_bias_diagnostic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()
    table = run(args.results, args.workers)
    nu = CONFIGS["noisy"].noise_dof
    print(f"theoretical E[log det C] - log det Sigma for one Wishart(nu={nu}) observation: {theoretical_bias(nu, 3):+.4f}")
    print(table.round(4).to_string(index=False))
    print(f"written to {args.results / 'summary' / 'noise_bias_diagnostic.csv'}")


if __name__ == "__main__":
    main()
