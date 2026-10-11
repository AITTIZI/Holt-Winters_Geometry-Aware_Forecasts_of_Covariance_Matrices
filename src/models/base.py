"""Interface and bookkeeping shared by the three Holt–Winters forecasters.

All models are the *same* additive (damped-)trend Holt–Winters recursion
written in the language of a Riemannian manifold (Exp, Log, parallel
transport); they differ only in the metric put on the SPD cone:

* Euclidean      -- the flat ambient metric of Sym(n) (classical element-wise HW);
* Log-Euclidean  -- the flat metric pulled back through the matrix logarithm;
* AIRM           -- the curved affine-invariant metric.

Subclasses implement ``_prepare`` (state initialisation, geometry-specific
pre-processing), ``_run`` (the recursion, vectorised over a batch of parameter
vectors) and ``_forecast_from_states``.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..geometry.spd import REPAIR_FLOOR, SPDViolationError, assert_spd, is_spd, repair_spd, sym
from ..optimization.parameter_fit import FitResult, ParameterSpace, fit_smoothing_parameters

PARAM_NAMES = ("alpha", "beta", "gamma", "phi")

LOSSES = ("mse", "huber")

FIT_METRICS = ("own", "airm", "frobenius")

DEFAULT_GRIDS = {
    "alpha": (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0),
    "beta": (0.0, 0.01, 0.03, 0.1, 0.2, 0.4),
    "gamma": (0.0, 0.05, 0.2, 0.5),
    "phi": (0.8, 0.9, 0.98),
}


@dataclass(frozen=True)
class HoltWintersSpec:

    trend: bool = True
    damped: bool = False
    season_length: int | None = None
    init_window: int = 10
    burn_in: int = 20
    alpha_bounds: tuple[float, float] = (0.01, 1.0)
    beta_bounds: tuple[float, float] = (0.0, 1.0)
    gamma_bounds: tuple[float, float] = (0.0, 1.0)
    phi_bounds: tuple[float, float] = (0.8, 0.98)
    loss: str = "mse"
    huber_k: float = 1.4
    fit_metric: str = "own"

    def __post_init__(self) -> None:
        if self.damped and not self.trend:
            raise ValueError("a damped trend requires trend=True")
        if self.season_length is not None and self.season_length < 2:
            raise ValueError("season_length must be >= 2 (or None)")
        if self.init_window < 1 or self.burn_in < 0:
            raise ValueError("init_window must be >= 1 and burn_in >= 0")
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}, got {self.loss!r}")
        if not self.huber_k > 0:
            raise ValueError("huber_k must be > 0")
        if self.fit_metric not in FIT_METRICS:
            raise ValueError(f"fit_metric must be one of {FIT_METRICS}, got {self.fit_metric!r}")

    @property
    def seasonal(self) -> bool:
        return self.season_length is not None

    @property
    def m(self) -> int:
        return self.season_length or 1

    @property
    def start(self) -> int:
        if not self.seasonal:
            return self.init_window
        m = self.season_length
        return m * math.ceil(max(self.init_window, 2 * m) / m)

    def fixed_parameters(self) -> dict[str, float]:
        fixed = {}
        if not self.trend:
            fixed["beta"] = 0.0
        if not self.seasonal:
            fixed["gamma"] = 0.0
        if not self.damped:
            fixed["phi"] = 1.0
        return fixed

    def free_parameters(self) -> tuple[str, ...]:
        names = ["alpha"]
        if self.trend:
            names.append("beta")
        if self.seasonal:
            names.append("gamma")
        if self.damped:
            names.append("phi")
        return tuple(names)

    def parameter_space(self) -> ParameterSpace:
        names = self.free_parameters()
        return ParameterSpace(
            names=names,
            bounds=tuple(getattr(self, f"{k}_bounds") for k in names),
            grids=tuple(np.array(DEFAULT_GRIDS[k], dtype=float) for k in names),
        )


@dataclass(frozen=True)
class SmoothingParams:
    alpha: float
    beta: float = 0.0
    gamma: float = 0.0
    phi: float = 1.0

    def as_array(self) -> np.ndarray:
        return np.array([self.alpha, self.beta, self.gamma, self.phi], dtype=float)


@dataclass
class FilterResult:

    params: SmoothingParams
    start: int
    one_step: np.ndarray
    level: np.ndarray
    error_sq: np.ndarray
    state: dict[str, np.ndarray]
    loss: float
    elapsed: float


def damped_horizon(phi: float | np.ndarray, horizons: np.ndarray) -> np.ndarray:
    """``phi_h = phi + phi^2 + ... + phi^h`` (equal to ``h`` when phi = 1)."""
    phi = np.asarray(phi, dtype=float)
    h = np.asarray(horizons, dtype=float)
    if np.all(phi == 1.0):
        return np.broadcast_to(h, np.broadcast_shapes(phi.shape, h.shape)).copy()
    return phi * (1.0 - phi**h) / (1.0 - phi)


def huber_terms(err_sq: np.ndarray, c: float) -> np.ndarray:
    with np.errstate(invalid="ignore", over="ignore"):
        d = np.sqrt(err_sq)
        return np.where(d <= c, err_sq, 2.0 * c * d - c * c)


def criterion_error_sq(F: np.ndarray, Y: np.ndarray, FY: np.ndarray, metric: str) -> np.ndarray:
    n = F.shape[-1]
    finite = np.isfinite(F).all(axis=(-2, -1))
    F = np.where(finite[:, None, None], F, np.eye(n))  # placeholders, scored inf below
    if metric == "frobenius":
        D = F - Y
        err_sq = np.einsum("bij,bij->b", D, D)
    elif metric == "airm":
        valid = is_spd(F)
        if not valid.all():
            F = np.where(valid[:, None, None], F, repair_spd(F, REPAIR_FLOOR))
        X = np.linalg.solve(FY, F)
        Z = sym(np.linalg.solve(FY, X.mT))
        w = np.linalg.eigvalsh(Z)
        with np.errstate(divide="ignore", invalid="ignore"):
            err_sq = np.where(np.all(w > 0, axis=-1), np.sum(np.log(np.where(w > 0, w, 1.0)) ** 2, axis=-1), np.inf)
    else:
        raise ValueError(f"unknown criterion {metric!r}")
    return np.where(finite, err_sq, np.inf)


def validate_series(Y: Any) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 3 or Y.shape[1] != Y.shape[2]:
        raise ValueError(f"expected an (N, n, n) array of matrices, got shape {Y.shape}")
    ok = is_spd(Y)
    if not np.all(ok):
        raise SPDViolationError(f"input series contains {np.count_nonzero(~ok)} non-SPD matrices")
    return Y


class HoltWintersModel(ABC):

    name: ClassVar[str]
    label: ClassVar[str]
    preserves_spd: ClassVar[bool]
    own_metric: ClassVar[str]

    def __init__(self, spec: HoltWintersSpec | None = None, params: SmoothingParams | None = None) -> None:
        self.spec = spec or HoltWintersSpec()
        self.params_ = params
        self.fit_result_: FitResult | None = None
        self.fit_seconds_: float | None = None
        self.loss_threshold_: float | None = None  

    
    def fit(self, Y: np.ndarray) -> "HoltWintersModel":
        Y = validate_series(Y)
        self._check_length(len(Y))
        prep = self._prepare(Y)
        space = self.spec.parameter_space()

        def batched_loss(th: np.ndarray) -> np.ndarray:
            return self._run(prep, self._expand(th, space.names), False)[0]

        t0 = time.perf_counter()
        self.loss_threshold_ = None
        result = fit_smoothing_parameters(batched_loss, space)
        if self.spec.loss == "huber":
            theta = self._expand(np.array([[result.params[k] for k in space.names]]), space.names)
            err_sq = self._run(prep, theta, True)[1]["error_sq"][self.spec.start + self.spec.burn_in :]
            scale = float(np.median(np.sqrt(err_sq)))
            if scale > 0: 
                self.loss_threshold_ = self.spec.huber_k * scale
                result = fit_smoothing_parameters(batched_loss, space)
        self.fit_seconds_ = time.perf_counter() - t0
        self.fit_result_ = result
        self.params_ = SmoothingParams(**result.params)
        return self

    def filter(self, Y: np.ndarray, params: SmoothingParams | None = None) -> FilterResult:

        params = params or self.params_
        if params is None:
            raise RuntimeError("model has no parameters: call fit() or pass params")

        clash = {k: getattr(params, k) for k, v in self.spec.fixed_parameters().items() if getattr(params, k) != v}
        if clash:
            raise ValueError(f"parameters {clash} contradict the specification {self.spec}")
        Y = validate_series(Y)
        self._check_length(len(Y))
        prep = self._prepare(Y)
        t0 = time.perf_counter()
        losses, parts = self._run(prep, params.as_array()[None, :], True)
        elapsed = time.perf_counter() - t0
        result = FilterResult(params=params, start=self.spec.start, loss=float(losses[0]), elapsed=elapsed, **parts)
        s = result.start
        if self.preserves_spd:
            assert_spd(result.one_step[s:], f"{self.name} one-step forecasts")
            assert_spd(result.level[s - 1 :], f"{self.name} levels")
        elif not np.all(np.isfinite(result.level[s - 1 :])):
            raise FloatingPointError(f"{self.name}: filter produced non-finite states")
        return result

    def forecast(self, result: FilterResult, origins: Any, horizons: Any) -> np.ndarray:

        origins = np.atleast_1d(np.asarray(origins, dtype=int))
        horizons = np.atleast_1d(np.asarray(horizons, dtype=int))
        N = len(result.level)
        if np.any(origins < result.start - 1) or np.any(origins >= N):
            raise ValueError("forecast origins must lie in [start - 1, N - 1]")
        if np.any(horizons < 1):
            raise ValueError("horizons must be >= 1")
        F = self._forecast_from_states(result, origins, horizons)
        if self.preserves_spd:
            assert_spd(F, f"{self.name} forecasts")
        return F

    def forecast_path(self, Y: np.ndarray, horizon: int) -> np.ndarray:

        result = self.filter(Y)
        return self.forecast(result, [len(Y) - 1], np.arange(1, horizon + 1))[0]

    # ------------------------------------------------------------ internals
    def _check_length(self, N: int) -> None:
        needed = self.spec.start + self.spec.burn_in + 2
        if N < needed:
            raise ValueError(f"series too short: need at least {needed} observations, got {N}")

    def _expand(self, theta: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
        full = np.tile(SmoothingParams(alpha=1.0).as_array(), (len(theta), 1))
        for j, name in enumerate(names):
            full[:, PARAM_NAMES.index(name)] = theta[:, j]
        return full

    @property
    def _criterion(self) -> str | None:
        metric = self.spec.fit_metric
        return None if metric in ("own", self.own_metric) else metric

    def _loss_count(self, N: int) -> int:
        return N - self.spec.start - self.spec.burn_in

    def _loss_terms(self, err_sq: np.ndarray) -> np.ndarray:
        c = self.loss_threshold_
        return err_sq if c is None else huber_terms(err_sq, c)

    @abstractmethod
    def _prepare(self, Y: np.ndarray) -> Any:
        """Parameter-independent pre-processing and initial state."""

    @abstractmethod
    def _run(self, prep: Any, theta: np.ndarray, store: bool) -> tuple[np.ndarray, dict | None]:
        """Recursion for a (B, 4) batch of parameters."""

    @abstractmethod
    def _forecast_from_states(self, result: FilterResult, origins: np.ndarray, horizons: np.ndarray) -> np.ndarray:
        """Multi-step forecasts from stored states."""
