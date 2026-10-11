from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .geometry.spd import REPAIR_FLOOR, SPDViolationError, is_spd, repair_spd, spd_factor, sym
from .metrics.evaluation import floor_tag, forecast_errors
from .models import MODEL_CLASSES, HoltWintersSpec

TAG = floor_tag(REPAIR_FLOOR)


def stein_loss(F: np.ndarray, C: np.ndarray) -> np.ndarray:

    F = np.asarray(F, dtype=float)
    finite = np.isfinite(F).all(axis=(-2, -1))
    F = np.where(finite[:, None, None], F, np.eye(F.shape[-1]))
    valid = is_spd(F)
    if not valid.all():
        F = np.where(valid[:, None, None], F, repair_spd(F, REPAIR_FLOOR))
    L = spd_factor(F)
    X = np.linalg.solve(L, C)
    z = np.linalg.eigvalsh(sym(np.linalg.solve(L, np.swapaxes(X, -1, -2))))
    loss = np.sum(z - np.log(z) - 1.0, axis=-1)
    return np.where(finite, loss, np.inf)


def origin_losses(F: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    e = forecast_errors(F, target, floors=(REPAIR_FLOOR,))
    return {
        "airm_sq": e[f"airm_{TAG}"] ** 2,
        "le_sq": e[f"le_{TAG}"] ** 2,
        "frob_sq": e["frob"] ** 2,
        "stein": stein_loss(F, target),
        "valid": e["valid"].astype(float),
    }


def refit_dates(T: int, window: int, refit_every: int) -> list[int]:
    return list(range(window, T, refit_every))


def rolling_block(C: np.ndarray, model_name: str, spec: HoltWintersSpec, tau: int, window: int, refit_every: int,
                  horizons: tuple[int, ...]) -> list[dict]:

    T = len(C)
    model = MODEL_CLASSES[model_name](spec).fit(C[tau - window : tau])
    end = min(tau + refit_every, T)
    start = tau - window
    result = model.filter(C[start:end])
    origins = np.arange(tau - 1, end - 1)
    p = model.params_
    rows = []
    for h in horizons:
        o = origins[origins + h <= T - 1]
        if not o.size:
            continue
        F = model.forecast(result, o - start, [h])[:, 0]
        losses = origin_losses(F, C[o + h])
        for k, origin in enumerate(o):
            rows.append({"model": model_name, "origin": int(origin), "horizon": h, "refit": tau,
                         "alpha": p.alpha, "beta": p.beta, "phi": p.phi, **{key: float(v[k]) for key, v in losses.items()}})
    return rows


def naive_block(C: np.ndarray, window: int, horizons: tuple[int, ...]) -> list[dict]:
    
    T = len(C)
    rows = []
    for h in horizons:
        o = np.arange(window - 1, T - h)
        losses = origin_losses(C[o], C[o + h])
        rows += [{"model": "naive", "origin": int(origin), "horizon": h, **{key: float(v[k]) for key, v in losses.items()}}
                 for k, origin in enumerate(o)]
    return rows


def newey_west_variance(d: np.ndarray, lag: int) -> float:
    d = np.asarray(d, dtype=float)
    x = d - d.mean()
    T = len(x)
    lrv = x @ x / T
    for k in range(1, lag + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1)) * (x[k:] @ x[:-k]) / T
    return float(lrv)


def dm_lag(T: int, h: int) -> int:
    return max(h - 1, int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))))


def diebold_mariano(loss_cand: np.ndarray, loss_ref: np.ndarray, h: int) -> dict[str, float]:

    d = np.asarray(loss_cand, dtype=float) - np.asarray(loss_ref, dtype=float)
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 3 or np.all(d == d[0]):
        return {"dm_stat": np.nan, "p_value": 1.0 if T >= 3 else np.nan, "n_origins": T, "lag": np.nan}
    lag = dm_lag(T, h)
    lrv = newey_west_variance(d, lag)
    stat = d.mean() / np.sqrt(lrv / T) if lrv > 0 else np.inf * np.sign(d.mean())
    stat *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return {"dm_stat": float(stat), "p_value": float(2.0 * student_t.sf(abs(stat), df=T - 1)), "n_origins": T, "lag": lag}


def block_bootstrap_ratio(num: np.ndarray, den: np.ndarray, rng: np.random.Generator, block: int = 12, n_boot: int = 2000,
                          root: bool = True, level: float = 0.95) -> tuple[float, float, float]:

    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    T = len(num)
    agg = (lambda x: np.sqrt(np.mean(x, axis=-1))) if root else (lambda x: np.mean(x, axis=-1))
    point = float(1.0 - agg(num) / agg(den))
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :T]
    boot = 1.0 - agg(num[idx]) / agg(den[idx])
    tail = 50.0 * (1.0 - level)
    lo, hi = np.percentile(boot, [tail, 100.0 - tail])
    return point, float(lo), float(hi)


LOSSES = {"airm_sq": True, "le_sq": True, "frob_sq": True, "stein": False}


def summarise(per_origin: pd.DataFrame, comparisons, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:

    from .metrics.statistics import holm_adjust

    rng = np.random.default_rng(seed)
    keys = ["dataset", "config"]
    summary_rows, test_rows = [], []
    for (dataset, config), g in per_origin.groupby(keys, sort=False):
        for (model, h), gm in g.groupby(["model", "horizon"], sort=False):
            rec = {"dataset": dataset, "config": config, "model": model, "horizon": h, "n_origins": len(gm),
                   "valid_ratio": gm["valid"].mean()}
            for loss, root in LOSSES.items():
                rec[loss.replace("_sq", "_rmse") if root else f"{loss}_mean"] = float(np.sqrt(gm[loss].mean()) if root else gm[loss].mean())
            for par in ("alpha", "beta", "phi"):
                if par in gm:
                    rec[par] = gm[par].mean()
            if "refit" in gm and gm["refit"].notna().any():
                refits = gm.drop_duplicates("refit")
                rec["n_refits"] = len(refits)
                rec["n_refits_trend"] = int((refits["beta"] > 1e-8).sum())  # re-estimations that fitted a trend
            summary_rows.append(rec)
        wide = {loss: g.pivot_table(index=["horizon", "origin"], columns="model", values=loss) for loss in LOSSES}
        for loss, root in LOSSES.items():
            family = []
            for cand, ref in comparisons:
                if cand not in wide[loss] or ref not in wide[loss]:
                    continue
                for h, w in wide[loss].groupby(level="horizon"):
                    a, b = w[cand].to_numpy(), w[ref].to_numpy()
                    point, lo, hi = block_bootstrap_ratio(a, b, rng, root=root)
                    family.append({"dataset": dataset, "config": config, "loss": loss, "candidate": cand, "reference": ref,
                                   "horizon": h, "rel_improvement": point, "rel_ci_low": lo, "rel_ci_high": hi,
                                   "win_share": float(np.mean(a < b)), **diebold_mariano(a, b, int(h))})
            if family:
                p_holm = holm_adjust(np.array([r["p_value"] for r in family]))
                for r, p in zip(family, p_holm):
                    r["p_holm"] = float(p)
                test_rows += family
    return pd.DataFrame(summary_rows), pd.DataFrame(test_rows)


REAL_CONFIGS = {
    "main": HoltWintersSpec(),
    "damped": HoltWintersSpec(damped=True),
    "ses": HoltWintersSpec(trend=False),
    "main_fitairm": HoltWintersSpec(fit_metric="airm"),
    "main_fitfrob": HoltWintersSpec(fit_metric="frobenius"),
    "damped_fitairm": HoltWintersSpec(damped=True, fit_metric="airm"),
    "damped_fitfrob": HoltWintersSpec(damped=True, fit_metric="frobenius"),
}


WINDOW, REFIT_EVERY, HORIZONS = 360, 12, (1, 3, 6, 12)

_DATA: dict[str, np.ndarray] = {}


def _init_worker(data: dict[str, np.ndarray]) -> None:
    _DATA.update(data)


def _block_task(task: tuple) -> list[dict]:
    dataset, config, model, tau, window, refit_every, horizons = task
    rows = rolling_block(_DATA[dataset], model, REAL_CONFIGS[config], tau, window, refit_every, horizons)
    return [{"dataset": dataset, "config": config, **r} for r in rows]


def run_rolling(data: dict[str, np.ndarray], workers: int, configs=tuple(REAL_CONFIGS), window: int = WINDOW,
                refit_every: int = REFIT_EVERY, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    import multiprocessing as mp

    tasks = [(name, config, model, tau, window, refit_every, horizons)
             for name, C in data.items() for config in configs for model in MODEL_CLASSES
             for tau in refit_dates(len(C), window, refit_every)]
    tasks.sort(key=lambda t: (t[2] != "airm", -data[t[0]].shape[-1]))  # longest tasks first
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(data,)) as pool:
        rows = [r for chunk in pool.imap_unordered(_block_task, tasks, chunksize=2) for r in chunk]
    for name, C in data.items():
        naive = naive_block(C, window, horizons)
        rows += [{"dataset": name, "config": config, **r} for config in configs for r in naive]
    return pd.DataFrame(rows).sort_values(["dataset", "config", "model", "horizon", "origin"]).reset_index(drop=True)


def series_features(name: str, C: np.ndarray, n_days: np.ndarray, months: np.ndarray) -> dict:
    from .geometry.airm import airm_distance

    w = np.linalg.eigvalsh(C)
    cond = w[:, -1] / w[:, 0]
    steps = airm_distance(C[:-1], C[1:])
    return {"dataset": name, "n": C.shape[-1], "n_months": len(C), "first_month": str(months[0]), "last_month": str(months[-1]),
            "min_days": int(n_days.min()), "median_days": float(np.median(n_days)), "max_days": int(n_days.max()),
            "cond_median": float(np.median(cond)), "cond_max": float(cond.max()),
            "eig_min": float(w.min()), "eig_max": float(w.max()),
            "step_median": float(np.median(steps)), "step_p99": float(np.percentile(steps, 99)),
            "first_origin_month": str(months[WINDOW - 1]), "n_refits": len(refit_dates(len(C), WINDOW, REFIT_EVERY))}


def audit_spd(per_origin: pd.DataFrame) -> int:
    geo = per_origin[per_origin["model"].isin([m for m, c in MODEL_CLASSES.items() if c.preserves_spd])]
    return int((geo["valid"] < 1).sum())


__all__ = [
    "REAL_CONFIGS",
    "SPDViolationError",
    "run_rolling",
    "series_features",
    "audit_spd",
    "block_bootstrap_ratio",
    "diebold_mariano",
    "dm_lag",
    "naive_block",
    "newey_west_variance",
    "origin_losses",
    "refit_dates",
    "rolling_block",
    "stein_loss",
    "summarise",
]
