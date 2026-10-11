"""Evaluation criteria: geodesic errors, SPD validity and paired statistics."""

from ..geometry.airm import airm_distance
from ..geometry.euclidean import frobenius_distance
from .evaluation import DEFAULT_FLOORS, floor_tag, forecast_errors, rmse, summarize_errors
from .statistics import bootstrap_mean_ci, holm_adjust, paired_comparison
