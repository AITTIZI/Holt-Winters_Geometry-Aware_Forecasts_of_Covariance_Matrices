"""Paired statistical comparison of forecasters across Monte Carlo replications.

Every model is evaluated on the *same* simulated paths, so comparisons are
paired: for each replication we take the difference in RMSE between two
models, test it with the Wilcoxon signed-rank test (no normality assumption),
and control the family-wise error rate with Holm's step-down procedure.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def bootstrap_mean_ci(x: np.ndarray, rng: np.random.Generator, n_boot: int = 2000, level: float = 0.95) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan"), float("nan")
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    tail = 50.0 * (1.0 - level)
    lo, hi = np.percentile(means, [tail, 100.0 - tail])
    return float(lo), float(hi)


def paired_comparison(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator | None = None,
    n_boot: int = 2000,
    level: float = 0.95,
) -> dict[str, float]:

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    out = {
        "n_pairs": int(ok.sum()),
        "median_diff": np.nan,
        "median_rel_improvement": np.nan,
        "rel_ci_low": np.nan,
        "rel_ci_high": np.nan,
        "win_rate": np.nan,
        "p_value": np.nan,
    }
    if a.size == 0:
        return out
    diff = a - b
    rel = 1.0 - a / b
    out["median_diff"] = float(np.median(diff))
    out["median_rel_improvement"] = float(np.median(rel))
    if rel.size >= 2:
        rng = np.random.default_rng(0) if rng is None else rng
        medians = np.median(rel[rng.integers(0, rel.size, size=(n_boot, rel.size))], axis=1)
        tail = 50.0 * (1.0 - level)
        out["rel_ci_low"], out["rel_ci_high"] = (float(q) for q in np.percentile(medians, [tail, 100.0 - tail]))
    out["win_rate"] = float(np.mean(a < b))
    if a.size >= 5 and np.any(diff != 0):
        out["p_value"] = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    elif a.size >= 5:
        out["p_value"] = 1.0
    return out
