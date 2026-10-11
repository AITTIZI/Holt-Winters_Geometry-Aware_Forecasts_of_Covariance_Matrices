from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

BatchedLoss = Callable[[np.ndarray], np.ndarray]

_PENALTY = 1e12

@dataclass(frozen=True)
class ParameterSpace:
    names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    grids: tuple[np.ndarray, ...]

    @property
    def dim(self) -> int:
        return len(self.names)

    def grid_points(self) -> np.ndarray:
        mesh = np.meshgrid(*self.grids, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=-1)


@dataclass(frozen=True)
class FitResult:
    params: dict[str, float]
    loss: float
    grid_loss: float
    n_evaluations: int
    optimizer_success: bool
    message: str
    n_starts: int = 1


def _finite_difference_points(x: np.ndarray, bounds: np.ndarray, h: float) -> tuple[np.ndarray, list]:
    pts, plan = [x], []
    for i in range(len(x)):
        lo, hi = bounds[i]
        up, down = x.copy(), x.copy()
        up[i] += h
        down[i] -= h
        if up[i] <= hi and down[i] >= lo:
            pts += [up, down]
            plan.append(("central", len(pts) - 2, len(pts) - 1))
        elif up[i] <= hi:
            pts.append(up)
            plan.append(("forward", len(pts) - 1, None))
        else:
            pts.append(down)
            plan.append(("backward", len(pts) - 1, None))
    return np.array(pts), plan


def grid_basins(losses: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:

    L = np.where(np.isfinite(losses), losses, np.inf).reshape(shape)
    basin = np.isfinite(L)
    for ax in range(L.ndim):
        pad = [(0, 0)] * L.ndim
        pad[ax] = (1, 1)
        P = np.pad(L, pad, constant_values=np.inf)
        lo, hi = [slice(None)] * L.ndim, [slice(None)] * L.ndim
        lo[ax], hi[ax] = slice(None, -2), slice(2, None)
        basin &= (L <= P[tuple(lo)]) & (L <= P[tuple(hi)])
    idx = np.flatnonzero(basin)
    idx = idx[np.argsort(L.ravel()[idx], kind="stable")]
    pos = np.stack(np.unravel_index(idx, shape), axis=-1)
    kept: list[int] = []
    for i in range(len(idx)):
        if all(np.max(np.abs(pos[i] - pos[k])) > 1 for k in kept):
            kept.append(i)
    return idx[kept]


def fit_smoothing_parameters(
    batched_loss: BatchedLoss,
    space: ParameterSpace,
    fd_step: float = 1e-5,
    maxiter: int = 100,
    n_starts: int = 1,
) -> FitResult:
    
    grid = space.grid_points()
    grid_losses = np.asarray(batched_loss(grid), dtype=float)
    n_evals = len(grid)
    finite = np.isfinite(grid_losses)
    if not np.any(finite):
        raise RuntimeError("every grid point produced a non-finite loss")
    starts = grid_basins(grid_losses, tuple(len(g) for g in space.grids))[:n_starts]
    x0, grid_loss = grid[starts[0]].copy(), float(grid_losses[starts[0]])
    bounds = np.array(space.bounds, dtype=float)

    def fun_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal n_evals
        pts, plan = _finite_difference_points(np.clip(x, bounds[:, 0], bounds[:, 1]), bounds, fd_step)
        L = np.asarray(batched_loss(pts), dtype=float)
        n_evals += len(pts)
        f = L[0] if np.isfinite(L[0]) else _PENALTY
        g = np.zeros(len(x))
        for i, (kind, a, b) in enumerate(plan):
            if kind == "central":
                g[i] = (L[a] - L[b]) / (2 * fd_step)
            elif kind == "forward":
                g[i] = (L[a] - L[0]) / fd_step
            else:
                g[i] = (L[0] - L[a]) / fd_step
        g[~np.isfinite(g)] = 0.0
        return float(f), g

    x, loss, res = x0, grid_loss, None
    for i in starts:
        r = minimize(
            fun_and_grad, grid[i], jac=True, method="L-BFGS-B", bounds=space.bounds, options={"maxiter": maxiter}
        )
        if res is None:
            res = r
        if r.fun < loss:  
            x, loss, res = np.clip(r.x, bounds[:, 0], bounds[:, 1]), float(r.fun), r
    return FitResult(
        params=dict(zip(space.names, map(float, x))),
        loss=loss,
        grid_loss=grid_loss,
        n_evaluations=n_evals,
        optimizer_success=bool(res.success),
        message=str(res.message),
        n_starts=len(starts),
    )
