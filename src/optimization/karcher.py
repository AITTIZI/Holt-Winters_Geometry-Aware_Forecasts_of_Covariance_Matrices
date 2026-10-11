from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.airm import le_mean
from ..geometry.spd import SPDViolationError, _require_positive, spd_factor, sym

_EPS = np.finfo(float).eps

ARMIJO = 1e-4


MIN_STEP = 1e-8



@dataclass(frozen=True)
class KarcherResult:
    mean: np.ndarray
    n_iter: int
    grad_norm: float
    status: str

    @property
    def converged(self) -> bool:
        return self.status == "converged"


def _cost(G: np.ndarray, FY: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    U, s, _ = np.linalg.svd(np.linalg.solve(G, FY))
    _require_positive(s, "karcher_mean (whitened sample)")
    logs = 2.0 * np.log(s)  
    cost = 0.5 * float(np.sum(w * np.sum(logs * logs, axis=-1)))
    return cost, sym(np.tensordot(w, (U * logs[:, None, :]) @ U.mT, axes=1))


def _trial_factor(G: np.ndarray, V: np.ndarray, lam: np.ndarray, step: float) -> np.ndarray:
    with np.errstate(over="ignore"):
        e = np.exp(0.5 * step * lam)
    _require_positive(e * e, "karcher_mean (trial step)")
    G_new = (G @ V) * e
    _require_positive(np.linalg.svd(G_new, compute_uv=False) ** 2, "karcher_mean (trial point)")
    return G_new


def karcher_mean(
    Y: np.ndarray,
    weights: np.ndarray | None = None,
    tol: float = 1e-12,
    max_iter: int = 200,
    init: np.ndarray | None = None,
) -> KarcherResult:
    Y = np.asarray(Y, dtype=float)
    w = np.full(len(Y), 1.0 / len(Y)) if weights is None else np.asarray(weights, float) / np.sum(weights)
    X = le_mean(Y, w) if init is None else np.array(init, dtype=float)
    FY = spd_factor(Y)

    G = spd_factor(X)
    cost, S = _cost(G, FY, w)
    grad_norm = float(np.linalg.norm(S))
    last_step = 1.0
    for it in range(1, max_iter + 1):
        if grad_norm <= tol:
            return KarcherResult(X, it - 1, grad_norm, "converged")
        lam, V = np.linalg.eigh(S)  
        step = min(1.0, 2.0 * last_step)
        while True:
            try:
                G_new = _trial_factor(G, V, lam, step)
                cost_new, S_new = _cost(G_new, FY, w)
            except SPDViolationError:
                cost_new = np.inf
            rounding = 64 * _EPS * max(cost, 1.0)
            margin = ARMIJO * step * grad_norm**2
            if margin > rounding:
                accept = cost_new <= cost - margin  
            else:

                accept = cost_new - cost <= rounding and np.linalg.norm(S_new) < grad_norm
            if accept:
                break
            step *= 0.5
            if step < MIN_STEP:  
                return KarcherResult(X, it - 1, grad_norm, "stalled")
        G, cost, S, last_step = G_new, cost_new, S_new, step
        X = sym(G @ G.T)
        grad_norm = float(np.linalg.norm(S))
    return KarcherResult(X, max_iter, grad_norm, "converged" if grad_norm <= tol else "max_iter")
