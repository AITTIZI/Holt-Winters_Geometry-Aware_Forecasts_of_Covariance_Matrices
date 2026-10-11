"""Riemannian Holt–Winters under the affine-invariant metric (AIRM).

State at time t: level ``L_t in SPD(n)``, trend ``T_t in T_{L_t}SPD(n)`` and
(optionally) m seasonal tangent vectors ``S_t^{(j)} in T_{L_t}SPD(n)``. Each
additive Holt–Winters update is replaced by its intrinsic counterpart:

    P_t     = Exp_{L_{t-1}}(phi T_{t-1})                  trend-projected level
    s_P     = Gamma_{L_{t-1} -> P_t}(S^{(t mod m)})         season moved to P_t
    Yhat_t  = Exp_{P_t}(s_P)                              one-step forecast
    L_t     = Exp_{P_t}(alpha [Log_{P_t}(Y_t) - s_P])      geodesic step toward the
                                                          deseasonalised observation
    T_t     = Gamma_{L_{t-1} -> L_t}(beta Log_{L_{t-1}}(L_t) + (1 - beta) phi T_{t-1})
    S_t^{(j)} = Gamma_{P_t -> L_t}(gamma Log_{P_t}(Y_t) + (1 - gamma) s_P)   (j = t mod m)
    other seasonal vectors are parallel-transported L_{t-1} -> L_t.

    Yhat_{t+h|t} = Exp_{Q_h}(Gamma_{L_t -> Q_h} S^{(t+h mod m)}),  Q_h = Exp_{L_t}(phi_h T_t).

Without seasonality the level update is the weighted geometric mean
``L_t = P_t #_alpha Y_t`` (a Riccati-type two-point mean). All operations are
congruence-equivariant, so the whole forecaster commutes with any change of
basis ``Y -> G Y G^T``; for commuting data it reduces to classical
Holt–Winters on the log-eigenvalues.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..geometry.airm import airm_log
from ..geometry.spd import SPDViolationError, spd_factor, sym
from ..optimization.karcher import karcher_mean
from .base import FilterResult, HoltWintersModel, criterion_error_sq, damped_horizon

_EPS = np.finfo(float).eps

REFINE_RATIO = 1e-8


@dataclass(frozen=True)
class _AIRMPrepared:
    Y: np.ndarray
    factor: np.ndarray  
    level0: np.ndarray  
    season0: np.ndarray | None  


def _gram_eigh(K: np.ndarray, tol: float, want_right: bool = False):
    """Eigendecomposition of the Gram matrix ``M = K K^T``, computed from its factor ``K``."""
    n = K.shape[-1]
    M = K @ K.mT
    finite = np.isfinite(M).all(axis=(-2, -1))
    if not finite.all():
        M = np.where(finite[:, None, None], M, np.eye(n))

    w, V = np.linalg.eigh(M)
    fast = finite & (w[:, 0] > REFINE_RATIO * w[:, -1])
    r = np.sqrt(w)
    logw = np.log(w)
    ok = fast.copy()
    Wt = (V.mT @ K) / r[:, :, None] if want_right else None
    refine = finite & ~fast
    if refine.any():

        U, s, Vh = np.linalg.svd(K[refine])
        U, s, Vh = U[..., ::-1], s[..., ::-1], Vh[..., ::-1, :]  
        V[refine], r[refine] = U, s
        with np.errstate(divide="ignore"):
            logw[refine] = 2.0 * np.log(s)
        ok[refine] = s[:, 0] > tol * s[:, -1]
        if want_right:
            Wt[refine] = Vh
    if not ok.all():
        bad = ~ok
        V[bad], r[bad], logw[bad] = np.eye(n), 1.0, 0.0
        if want_right:
            Wt[bad] = np.eye(n)
    return V, r, logw, ok, Wt


def _checked_exp(x: np.ndarray, fail, what: str) -> np.ndarray:

    e = np.exp(x)
    ok = np.all((e > 0.0) & (e < np.inf), axis=-1)
    if not ok.all():
        fail(~ok, what)
        e[~ok] = 1.0
    return e


def _numerically_spd(F: np.ndarray, tol: float) -> np.ndarray:

    A = F @ F.mT
    finite = np.isfinite(A).all(axis=(-2, -1))
    if not finite.all():
        A = np.where(finite[:, None, None], A, np.eye(A.shape[-1]))
    w = np.linalg.eigvalsh(A)
    return finite & (w[:, 0] > tol * w[:, -1])


INIT_GRAD_TOL = 1e-6

def _initial_mean(Y: np.ndarray) -> np.ndarray:
    result = karcher_mean(Y)
    if result.grad_norm > INIT_GRAD_TOL:
        warnings.warn(
            f"Karcher mean for AIRM initialisation is inaccurate ({result.status}, gradient norm {result.grad_norm:.2e})",
            RuntimeWarning,
            stacklevel=3,
        )
    return result.mean


class AIRMHoltWinters(HoltWintersModel):
    name = "airm"
    label = "AIRM HW"
    preserves_spd = True
    own_metric = "airm"

    def _prepare(self, Y: np.ndarray) -> _AIRMPrepared:
        k, m = self.spec.start, self.spec.m
        level0 = _initial_mean(Y[:k])
        G0 = spd_factor(level0)
        season0 = None
        if self.spec.seasonal:
            S0 = np.stack([airm_log(level0, _initial_mean(Y[j:k:m])) for j in range(m)])
            S0 -= S0.mean(axis=0)
            season0 = sym(np.linalg.solve(G0, np.linalg.solve(G0, S0).mT))

        return _AIRMPrepared(Y, spd_factor(Y), G0, season0)

    def _run(self, prep: _AIRMPrepared, theta: np.ndarray, store: bool) -> tuple[np.ndarray, dict | None]:
        Y, FY = prep.Y, prep.factor
        N, n, _ = Y.shape
        B = len(theta)
        alpha, beta, gamma, phi = (theta[:, i, None] for i in range(4))  
        start, burn_in, m, seasonal = self.spec.start, self.spec.burn_in, self.spec.m, self.spec.seasonal
        tol = n * _EPS
        eye = np.eye(n)

        check = not store

        trend_free = not seasonal and not np.any(theta[:, 1])
        criterion = self._criterion  

        G = np.broadcast_to(prep.level0, (B, n, n)).copy()  
        tau = np.zeros((B, n, n))  
        S = np.broadcast_to(prep.season0, (B, m, n, n)).copy() if seasonal else None
        alive = np.ones(B, dtype=bool)  
        loss_ok = np.ones(B, dtype=bool)  
        sse = np.zeros(B)

        if store:
            hist_G = np.full((N, n, n), np.nan)
            hist_tau = np.full((N, n, n), np.nan)
            hist_S = np.full((N, m, n, n), np.nan) if seasonal else None
            hist_F = np.full((N, n, n), np.nan) 
            hist_err = np.full(N, np.nan)
            hist_G[start - 1], hist_tau[start - 1] = G[0], tau[0]
            if seasonal:
                hist_S[start - 1] = S[0]

        def fail(bad: np.ndarray, what: str) -> None:
            if store:
                raise SPDViolationError(f"airm: filter state left the SPD manifold ({what})")
            alive[bad] = False

        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            for t in range(start, N):
                if trend_free:
                    GP = GF = G  
                else:

                    wt, Vt = np.linalg.eigh(tau)
                    e = _checked_exp(0.5 * phi * wt, fail, "trend extrapolation")
                    GP = (G @ Vt) * e[:, None, :]
                    if seasonal:
                        j = t % m
                        sig = Vt.mT @ S[:, j] @ Vt  
                        ws, Vs = np.linalg.eigh(sig)
                        es = _checked_exp(0.5 * ws, fail, "seasonal component")
                        GF = (GP @ Vs) * es[:, None, :]  
                    else:
                        GF = GP

                if criterion is not None:

                    crit_err_sq = criterion_error_sq(GF @ GF.mT, Y[t], FY[t], criterion)

                if check:

                    if GF is G:
                        ok = _numerically_spd(G, tol)
                    else:
                        ok = _numerically_spd(np.concatenate((G, GF)), tol).reshape(2, B).all(axis=0)
                    if not ok.all():
                        fail(~ok, "prediction")

                C = np.linalg.solve(GP, FY[t])
                Vz, _, lz, ok, _ = _gram_eigh(C, tol)
                if not ok.all():
                    fail(~ok, "whitened observation")

                if trend_free:
                    err_sq = np.sum(lz**2, axis=-1)

                    G = (G @ Vz) * _checked_exp(0.5 * alpha * lz, fail, "level step")[:, None, :]
                else:
                    if seasonal:

                        C2 = (Vs.mT @ C) / es[:, :, None]
                        _, _, lr, ok2, _ = _gram_eigh(C2, tol)
                        loss_ok &= ok2
                        err_sq = np.sum(lr**2, axis=-1)
                        omega = (Vz * lz[:, None, :]) @ Vz.mT  
                        wd, Vd = np.linalg.eigh(alpha[:, :, None] * (omega - sig))
                        K = ((Vt * e[:, None, :]) @ Vd) * np.exp(0.5 * wd)[:, None, :]
                    else:
                        err_sq = np.sum(lz**2, axis=-1)
                        K = ((Vt * e[:, None, :]) @ Vz) * np.exp(0.5 * alpha * lz)[:, None, :]

                    Vm, rm, lm, ok, Wt = _gram_eigh(K, tol, want_right=seasonal)
                    if not ok.all():
                        fail(~ok, "level step")

                    Q = Vt.mT @ Vm
                    tau = (1.0 - beta[:, :, None]) * phi[:, :, None] * ((Q.mT * wt[:, None, :]) @ Q)
                    tau += (beta * lm)[:, :, None] * eye
                    if seasonal:
                        y = gamma[:, :, None] * omega + (1.0 - gamma[:, :, None]) * sig
                        R = Wt @ Vd.mT
                        s_new = R @ y @ R.mT
                        S = Vm.mT[:, None] @ S @ Vm[:, None]
                        S[:, j] = s_new
                    G = (G @ Vm) * rm[:, None, :]

                if not alive.all():
                    dead = ~alive
                    G[dead], tau[dead] = eye, 0.0
                    if seasonal:
                        S[dead] = 0.0

                if criterion is not None:
                    err_sq = crit_err_sq
                if t >= start + burn_in:
                    sse += self._loss_terms(err_sq)
                if store:
                    hist_F[t], hist_G[t], hist_tau[t], hist_err[t] = GF[0], G[0], tau[0], err_sq[0]
                    if seasonal:
                        hist_S[t] = S[0]

        if check:
            alive &= _numerically_spd(G, tol)  # final level L_{N-1}
        loss = np.where(alive & loss_ok & np.isfinite(sse), sse / self._loss_count(N), np.inf)
        if not store:
            return loss, None

        level = sym(hist_G @ hist_G.mT)
        state = {
            "level": level,
            "trend": sym(hist_G @ hist_tau @ hist_G.mT),
            "level_factor": hist_G,
            "trend_coords": sym(hist_tau),
        }
        if seasonal:
            state["season"] = sym(hist_G[:, None] @ hist_S @ hist_G[:, None].mT)
            state["season_coords"] = sym(hist_S)
        return loss, {"one_step": sym(hist_F @ hist_F.mT), "level": level, "error_sq": hist_err, "state": state}

    def _forecast_from_states(self, result: FilterResult, origins: np.ndarray, horizons: np.ndarray) -> np.ndarray:
        F = self._forecast_factors(result, origins, horizons)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            return sym(F @ F.mT)

    def _forecast_factors(self, result: FilterResult, origins: np.ndarray, horizons: np.ndarray) -> np.ndarray:
        """Factors ``F`` of the h-step forecasts ``F F^T``, shape (O, H, n, n)."""
        G = result.state["level_factor"][origins]
        wt, Vt = np.linalg.eigh(result.state["trend_coords"][origins])
        phi_h = damped_horizon(result.params.phi, horizons)[None, :, None]
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            F = (G @ Vt)[:, None] * np.exp(0.5 * phi_h * wt[:, None, :])[:, :, None, :]
            if self.spec.seasonal:
                phases = (origins[:, None] + horizons[None, :]) % self.spec.m
                s = result.state["season_coords"][origins[:, None], phases]
                ws, Vs = np.linalg.eigh(Vt.mT[:, None] @ s @ Vt[:, None])
                F = (F @ Vs) * np.exp(0.5 * ws)[..., None, :]
        return F
