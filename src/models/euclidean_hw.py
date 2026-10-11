"""Holt–Winters in a flat chart, and the classical Euclidean (element-wise) model.

For a flat geometry with a global chart ``c: SPD(n) -> Sym(n)`` in which the
metric is Frobenius, Exp/Log/parallel transport are vector-space operations in
chart coordinates ``X_t = c(Y_t)``. The Riemannian Holt–Winters recursion
therefore reduces to the textbook additive damped-trend model:

    l_t = alpha (X_t - s_{t-m}) + (1 - alpha)(l_{t-1} + phi b_{t-1})
    b_t = beta (l_t - l_{t-1}) + (1 - beta) phi b_{t-1}
    s_t = gamma (X_t - l_{t-1} - phi b_{t-1}) + (1 - gamma) s_{t-m}
    Xhat_{t+h|t} = l_t + phi_h b_t + s_{t+h-m(k+1)},     Yhat = c^{-1}(Xhat).

"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from ..geometry.spd import spd_factor, sym
from .base import FilterResult, HoltWintersModel, criterion_error_sq, damped_horizon


@dataclass(frozen=True)
class _FlatPrepared:
    X: np.ndarray
    level0: np.ndarray
    season0: np.ndarray | None
    Y: np.ndarray | None = None  
    factor: np.ndarray | None = None  


class FlatHoltWinters(HoltWintersModel):

    @abstractmethod
    def _to_chart(self, Y: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def _from_chart(self, X: np.ndarray) -> np.ndarray: ...

    def _to_ambient(self, X: np.ndarray) -> np.ndarray:
        return self._from_chart(X)

    def _representable(self, X: np.ndarray) -> np.ndarray:
        return np.ones(len(X), dtype=bool)

    def _prepare(self, Y: np.ndarray) -> _FlatPrepared:
        X = self._to_chart(Y)
        k, m = self.spec.start, self.spec.m
        level0 = X[:k].mean(axis=0) 
        season0 = None
        if self.spec.seasonal:
            season0 = np.stack([X[j:k:m].mean(axis=0) - level0 for j in range(m)])
            season0 -= season0.mean(axis=0)
        criterion = self._criterion
        return _FlatPrepared(
            X, level0, season0,
            Y=Y if criterion else None,
            factor=spd_factor(Y) if criterion == "airm" else None,
        )

    def _run(self, prep: _FlatPrepared, theta: np.ndarray, store: bool) -> tuple[np.ndarray, dict | None]:
        X = prep.X
        N, n, _ = X.shape
        B = len(theta)
        alpha, beta, gamma, phi = (theta[:, i, None, None] for i in range(4))
        start, burn_in, m, seasonal = self.spec.start, self.spec.burn_in, self.spec.m, self.spec.seasonal

        criterion = self._criterion

        check = self.preserves_spd and not store

        level = np.broadcast_to(prep.level0, (B, n, n)).copy()
        trend = np.zeros((B, n, n))
        season = np.broadcast_to(prep.season0, (B, m, n, n)).copy() if seasonal else None
        sse = np.zeros(B)
        alive = np.ones(B, dtype=bool)

        if store:
            hist_level = np.full((N, n, n), np.nan)
            hist_trend = np.full((N, n, n), np.nan)
            hist_season = np.full((N, m, n, n), np.nan) if seasonal else None
            hist_pred = np.full((N, n, n), np.nan)
            hist_err = np.full(N, np.nan)
            hist_level[start - 1], hist_trend[start - 1] = level[0], trend[0]
            if seasonal:
                hist_season[start - 1] = season[0]

        with np.errstate(over="ignore", invalid="ignore"):
            for t in range(start, N):
                projected = level + phi * trend
                if seasonal:
                    j = t % m
                    s_old = season[:, j]
                    pred = projected + s_old
                else:
                    pred = projected
                if check: 
                    alive &= self._representable(np.concatenate((level, pred))).reshape(2, B).all(axis=0)
                err = X[t] - pred
                if criterion is None:
                    err_sq = np.einsum("bij,bij->b", err, err)
                else:
                    err_sq = criterion_error_sq(self._to_ambient(pred), prep.Y[t], None if prep.factor is None else prep.factor[t], criterion)
                if t >= start + burn_in:
                    sse += self._loss_terms(err_sq)
                new_level = projected + alpha * err
                trend = beta * (new_level - level) + (1.0 - beta) * phi * trend
                if seasonal:
                    season[:, j] = s_old + gamma * err
                level = new_level
                if store:
                    hist_pred[t], hist_level[t], hist_trend[t], hist_err[t] = pred[0], level[0], trend[0], err_sq[0]
                    if seasonal:
                        hist_season[t] = season[0]

        if check:
            alive &= self._representable(level) 
        loss = sse / self._loss_count(N)
        loss[~np.isfinite(loss) | ~alive] = np.inf
        if not store:
            return loss, None

        one_step = np.full((N, n, n), np.nan)
        one_step[start:] = self._from_chart(hist_pred[start:])
        level_spd = np.full((N, n, n), np.nan)
        level_spd[start - 1 :] = self._from_chart(hist_level[start - 1 :])
        state = {"level": hist_level, "trend": hist_trend}
        if seasonal:
            state["season"] = hist_season
        return loss, {"one_step": one_step, "level": level_spd, "error_sq": hist_err, "state": state}

    def _forecast_from_states(self, result: FilterResult, origins: np.ndarray, horizons: np.ndarray) -> np.ndarray:
        level = result.state["level"][origins][:, None]
        trend = result.state["trend"][origins][:, None]
        phi_h = damped_horizon(result.params.phi, horizons)[None, :, None, None]
        X = level + phi_h * trend
        if self.spec.seasonal:
            phases = (origins[:, None] + horizons[None, :]) % self.spec.m
            X = X + result.state["season"][origins[:, None], phases]
        return self._from_chart(X)


class EuclideanHoltWinters(FlatHoltWinters):
    name = "euclidean"
    label = "Euclidean HW"
    preserves_spd = False
    own_metric = "frobenius"

    def _to_chart(self, Y: np.ndarray) -> np.ndarray:
        return np.array(Y, dtype=float)

    def _from_chart(self, X: np.ndarray) -> np.ndarray:
        return sym(X)
