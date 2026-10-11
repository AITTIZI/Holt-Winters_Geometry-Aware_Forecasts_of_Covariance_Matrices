"""Forecast-accuracy and validity metrics for SPD-matrix forecasts."""

from __future__ import annotations
import numpy as np
from ..geometry.airm import airm_distance, le_distance
from ..geometry.euclidean import frobenius_distance
from ..geometry.spd import REPAIR_FLOOR, is_spd, repair_spd

DEFAULT_FLOORS = (1e-2, 1e-3, 1e-4, 1e-6)
MEAN_KEYS = ("trace_rel", "logdet_abs", "log_lmax_ratio", "logdet_err", "logtr_err")



def floor_tag(floor: float) -> str:
    return f"rep{floor:.0e}"


def rmse(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    return float(np.sqrt(np.mean(d**2))) if d.size else float("nan")


def forecast_errors(F: np.ndarray, target: np.ndarray, floors: tuple[float, ...] = DEFAULT_FLOORS) -> dict[str, np.ndarray]:
    F = np.asarray(F, dtype=float)
    finite = np.all(np.isfinite(F), axis=(-2, -1))
    valid = is_spd(F)  # already False for non-finite forecasts
    w_f = np.full(F.shape[:-1], np.nan)
    if np.any(finite):
        w_f[finite] = np.linalg.eigvalsh(F[finite])
    w_t = np.linalg.eigvalsh(target)
    trace_t = np.trace(target, axis1=-2, axis2=-1)
    out: dict[str, np.ndarray] = {
        "valid": valid,
        "min_eig": w_f[:, 0],
        "frob": np.where(finite, frobenius_distance(np.where(finite[:, None, None], F, 0.0), target), np.inf),
        "airm_valid": np.full(len(F), np.nan),
        "le_valid": np.full(len(F), np.nan),
        "trace_rel": np.where(finite, np.abs(np.trace(np.where(finite[:, None, None], F, 0.0), axis1=-2, axis2=-1) - trace_t) / trace_t, np.inf),
        "logdet_abs": np.full(len(F), np.nan),
        "logdet_err": np.full(len(F), np.nan),
        "log_lmax_ratio": np.full(len(F), np.nan),
        "logtr_err": np.full(len(F), np.nan),
    }
    trace_f = np.trace(np.where(finite[:, None, None], F, 0.0), axis1=-2, axis2=-1)
    pos_trace = finite & (trace_f > 0)
    out["logtr_err"][pos_trace] = np.log(trace_f[pos_trace] / trace_t[pos_trace])
    positive_top = finite & (np.nan_to_num(w_f[:, -1], nan=-1.0) > 0)
    out["log_lmax_ratio"][positive_top] = np.log(w_f[positive_top, -1] / w_t[positive_top, -1])
    if np.any(valid):
        out["airm_valid"][valid] = airm_distance(F[valid], target[valid])
        out["le_valid"][valid] = le_distance(F[valid], target[valid])
        out["logdet_err"][valid] = np.sum(np.log(w_f[valid]), axis=-1) - np.sum(np.log(w_t[valid]), axis=-1)
        out["logdet_abs"][valid] = np.abs(out["logdet_err"][valid])
    repairable = ~valid & finite
    for floor in floors:
        tag = floor_tag(floor)
        airm = out["airm_valid"].copy()
        if np.any(repairable):
            airm[repairable] = airm_distance(repair_spd(F[repairable], floor), target[repairable])
        airm[~finite] = np.inf
        out[f"airm_{tag}"] = airm
    le = out["le_valid"].copy()
    if np.any(repairable):
        le[repairable] = le_distance(repair_spd(F[repairable], REPAIR_FLOOR), target[repairable])
    le[~finite] = np.inf
    out[f"le_{floor_tag(REPAIR_FLOOR)}"] = le
    return out


def summarize_errors(errors: dict[str, np.ndarray]) -> dict[str, float]:
    min_eig = errors["min_eig"]
    summary = {
        "n_forecasts": int(len(errors["valid"])),
        "n_nonfinite": int(np.count_nonzero(np.isnan(min_eig))),
        "valid_ratio": float(np.mean(errors["valid"])),
        "min_eig": float(np.nanmin(min_eig)) if np.any(~np.isnan(min_eig)) else float("nan"),
    }
    for key, values in errors.items():
        if key in MEAN_KEYS:
            defined = values[~np.isnan(values)]  # NaN = undefined by design; inf propagates
            summary[f"{key}_mean"] = float(defined.mean()) if defined.size else float("nan")
        elif key not in ("valid", "min_eig"):
            summary[f"{key}_rmse"] = rmse(values)
    return summary
