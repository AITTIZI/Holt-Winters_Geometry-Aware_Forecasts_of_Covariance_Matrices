"""Log-Euclidean Holt–Winters: the Euclidean recursion in the chart ``X = log C``.

The Log-Euclidean (LE) metric (Arsigny et al. 2007) is the pull-back of the
Frobenius metric through the principal matrix logarithm, a global
diffeomorphism ``SPD(n) -> Sym(n)``. The geometry is flat, so (module
docstring of :mod:`.euclidean_hw`) the Riemannian Holt–Winters recursion is the
textbook recursion on ``X_t = log C_t``, mapped back by ``exp``:

    l_t = alpha (X_t - s_{t-m}) + (1 - alpha)(l_{t-1} + phi b_{t-1}),   ...
    Yhat_{t+h|t} = exp(l_t + phi_h b_t + s_{t+h-m(k+1)}).

* Fitting loss (own geometry): ``d_LE(A, B)^2 = ||log A - log B||_F^2``.
* Initial level: the LE Fréchet mean of the first ``k`` observations,
  ``exp(mean_j log C_j)``: the "mean in its own geometry" rule shared by all models.
* Forecasts: ``exp`` of a symmetric matrix, hence SPD at every horizon by
  construction. No eigenvalue is ever clipped.
"""

from __future__ import annotations

import numpy as np

from ..geometry.spd import expm_sym, from_eig, logm_spd, positive_mask
from .euclidean_hw import FlatHoltWinters

_EPS = np.finfo(float).eps


def _exp_spectrum(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    w, V = np.linalg.eigh(X)
    with np.errstate(over="ignore"):
        ew = np.exp(w)
    return V, ew, positive_mask(ew)


class LogEuclideanHoltWinters(FlatHoltWinters):
    name = "le"
    label = "Log-Euclidean HW"
    preserves_spd = True
    own_metric = "le"

    def _to_chart(self, Y: np.ndarray) -> np.ndarray:
        return logm_spd(Y)

    def _from_chart(self, X: np.ndarray) -> np.ndarray:
        return expm_sym(X)  

    def _to_ambient(self, X: np.ndarray) -> np.ndarray:
        V, ew, ok = _exp_spectrum(X)
        F = from_eig(np.where(ok[:, None], ew, 1.0), V)
        F[~ok] = np.nan  
        return F

    def _representable(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[-1]
        finite = np.isfinite(X).all(axis=(-2, -1))
        X = np.nan_to_num(X)
        centred = X - (np.trace(X, axis1=-2, axis2=-1) / n)[:, None, None] * np.eye(n)
        spread_bound = np.sqrt(2.0) * np.linalg.norm(centred, axis=(-2, -1))
        ok = finite & (spread_bound < -np.log(n * _EPS) - 1.0) & (np.linalg.norm(X, axis=(-2, -1)) < 700.0)
        rest = finite & ~ok
        if rest.any():
            ok[rest] = _exp_spectrum(X[rest])[2]
        return ok
