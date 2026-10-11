"""Aggregation of per-replication metrics into tables and paired tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..geometry.spd import REPAIR_FLOOR
from .evaluation import floor_tag
from .statistics import bootstrap_mean_ci, holm_adjust, paired_comparison

PRIMARY_FLOOR = REPAIR_FLOOR
PRIMARY_METRIC = f"airm_{floor_tag(PRIMARY_FLOOR)}_rmse"
LE_METRIC = f"le_{floor_tag(PRIMARY_FLOOR)}_rmse"
COMPARISONS = (("airm", "le"), ("airm", "euclidean"), ("le", "euclidean"))
KEYS = ["config", "scenario", "model", "horizon"]


def summary_table(df: pd.DataFrame, metric: str = PRIMARY_METRIC, seed: int = 0) -> pd.DataFrame:

    ok = df[df["status"] == "ok"]
    rng = np.random.default_rng(seed)
    records = []
    for key, g in ok.groupby(KEYS, sort=False):
        lo, hi = bootstrap_mean_ci(g[metric].to_numpy(), rng)
        records.append(
            {
                **dict(zip(KEYS, key)),
                "n_reps": len(g),
                "mean": g[metric].mean(),
                "ci_low": lo,
                "ci_high": hi,
                "median": g[metric].median(),
                "valid_ratio": g["valid_ratio"].mean(),
                "reps_with_invalid": int((g["valid_ratio"] < 1).sum()),
                "airm_valid_rmse": g["airm_valid_rmse"].mean(),
                "frob_rmse": g["frob_rmse"].mean(),
                "le_rmse": g[LE_METRIC].mean() if LE_METRIC in g else np.nan,
                "level_valid_ratio": g["level_valid_ratio"].mean(),
                "trace_rel": g["trace_rel_mean"].mean(),
                "logdet_abs": g["logdet_abs_mean"].mean(),
                "log_lmax_ratio": g["log_lmax_ratio_mean"].mean(),
                "logdet_err": g["logdet_err_mean"].mean() if "logdet_err_mean" in g else np.nan,
                "logtr_err": g["logtr_err_mean"].mean() if "logtr_err_mean" in g else np.nan,
                "alpha": g["alpha"].mean(),
                "beta": g["beta"].mean(),
                "phi": g["phi"].mean(),
                "fit_seconds": g["fit_seconds"].median(),
                "filter_us_per_step": g["filter_us_per_step"].median(),
            }
        )
    errors = df[df["status"] != "ok"].groupby(KEYS, sort=False).size().rename("n_errors").reset_index()
    out = pd.DataFrame.from_records(records)
    out = out.merge(errors, on=KEYS, how="left") if len(errors) else out.assign(n_errors=0)
    out["n_errors"] = out["n_errors"].fillna(0).astype(int)
    out.attrs["metric"] = metric
    return out


def pairwise_tests(df: pd.DataFrame, metric: str = PRIMARY_METRIC, seed: int = 0) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    ok = df[df["status"] == "ok"]
    wide = ok.pivot_table(index=["config", "scenario", "horizon", "rep"], columns="model", values=metric)
    records = []
    for (config, scenario, horizon), g in wide.groupby(level=["config", "scenario", "horizon"], sort=False):
        for cand, ref in COMPARISONS:
            if cand in g and ref in g:
                a, b = g[cand].to_numpy(), g[ref].to_numpy()
                stats = paired_comparison(a, b, rng)
                both = np.isfinite(a) & np.isfinite(b)
                max_abs_diff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else np.nan  # 0: identical forecasters
                records.append({"config": config, "scenario": scenario, "horizon": horizon, "candidate": cand, "reference": ref,
                                **stats, "max_abs_diff": max_abs_diff})
    out = pd.DataFrame.from_records(records)
    out["p_holm"] = np.nan
    for _, idx in out.groupby("config").groups.items():
        out.loc[idx, "p_holm"] = holm_adjust(out.loc[idx, "p_value"].to_numpy())
    out.attrs["metric"] = metric
    return out


def params_summary(df: pd.DataFrame) -> pd.DataFrame:

    ok = df[(df["status"] == "ok") & (df["horizon"] == df["horizon"].min())]
    g = ok.groupby(["config", "scenario", "model"], sort=False)
    return pd.DataFrame({
        "n_fits": g.size(), "alpha": g["alpha"].mean(), "beta": g["beta"].mean(),
        "share_trend": g["beta"].apply(lambda b: float(np.mean(b > 1e-8))), "phi": g["phi"].mean(),
        "share_alpha_one": g["alpha"].apply(lambda a: float(np.mean(a > 1 - 1e-8))),
    }).reset_index()


def floor_sensitivity(df: pd.DataFrame, floors: tuple[float, ...]) -> pd.DataFrame:

    ok = df[df["status"] == "ok"]
    e = ok[ok["model"] == "euclidean"]
    cols = {f"floor {f:g}": f"airm_{floor_tag(f)}_rmse" for f in floors}
    agg = e.groupby(["config", "scenario", "horizon"], sort=False).agg(
        valid_ratio=("valid_ratio", "mean"), valid_only=("airm_valid_rmse", "mean"), **{k: (v, "mean") for k, v in cols.items()}
    )
    for model in ("le", "airm"):
        ref = ok[ok["model"] == model].groupby(["config", "scenario", "horizon"], sort=False)[PRIMARY_METRIC].mean()
        agg[model] = ref.reindex(agg.index)
    return agg.reset_index()


def volume_bias(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    ok = df[df["status"] == "ok"]
    records = []
    for key, g in ok.groupby(KEYS, sort=False):
        rec = {**dict(zip(KEYS, key)), "n_reps": len(g), "valid_ratio": g["valid_ratio"].mean()}
        for col in ("logdet_err", "logtr_err"):
            x = g[f"{col}_mean"].to_numpy(dtype=float)
            lo, hi = bootstrap_mean_ci(x, rng)
            rec |= {col: float(np.nanmean(x)) if np.isfinite(x).any() else np.nan, f"{col}_lo": lo, f"{col}_hi": hi,
                    f"{col}_reps": int(np.isfinite(x).sum())}
        records.append(rec)
    return pd.DataFrame.from_records(records)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "nan" if not np.isfinite(v) else format(v, floatfmt)
        return str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])
