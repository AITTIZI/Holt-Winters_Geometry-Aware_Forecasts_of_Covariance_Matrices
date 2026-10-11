"""Holt–Winters forecasters for SPD-matrix time series under three geometries."""

from .airm_hw import AIRMHoltWinters
from .base import (
    FIT_METRICS,
    FilterResult,
    HoltWintersModel,
    HoltWintersSpec,
    SmoothingParams,
    criterion_error_sq,
    damped_horizon,
    validate_series,
)
from .euclidean_hw import EuclideanHoltWinters, FlatHoltWinters
from .log_euclidean_hw import LogEuclideanHoltWinters

MODEL_CLASSES: dict[str, type[HoltWintersModel]] = {
    cls.name: cls for cls in (EuclideanHoltWinters, LogEuclideanHoltWinters, AIRMHoltWinters)
}

