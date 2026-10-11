from __future__ import annotations

import numpy as np

from .spd import (
    _require_positive,
    expm_sym,
    logm_spd,
    spd_factor,
    sqrt_invsqrt_spd,
    sym,
    transpose,
)

REFINE_RATIO = 1e-8

def _whiten(F: np.ndarray, B: np.ndarray) -> np.ndarray:
    X = np.linalg.solve(F, B)
    return sym(np.linalg.solve(F, transpose(X)))


def _whitened_eigh(A: np.ndarray, B: np.ndarray, vectors: bool = True):

    A, B = np.broadcast_arrays(np.asarray(A, float), np.asarray(B, float))
    F = spd_factor(sym(A))  # raises SPDViolationError when A is not positive definite
    Z = _whiten(F, sym(B))
    if vectors:
        w, U = np.linalg.eigh(Z)
    else:
        w, U = np.linalg.eigvalsh(Z), None
    n = w.shape[-1]
    with np.errstate(invalid="ignore"):
        refine = ~(w[..., 0] > REFINE_RATIO * w[..., -1])  
    if not np.any(refine):
        return F, U, np.log(w)
    shape = w.shape
    w, refine = w.reshape(-1, n), refine.reshape(-1)
    Fr = F.reshape(-1, n, n)[refine]
    K = np.linalg.solve(Fr, spd_factor(B.reshape(-1, n, n)[refine])) 
    logw = np.log(np.where(refine[:, None], 1.0, w))
    if vectors:
        Uk, s, _ = np.linalg.svd(K)
        Uk, s = Uk[..., ::-1], s[..., ::-1]  
        U = U.reshape(-1, n, n).copy()
        U[refine] = Uk
        U = U.reshape(*shape, n)
    else:
        s = np.linalg.svd(K, compute_uv=False)[:, ::-1]
    _require_positive(s, "AIRM whitening (second argument)")
    logw[refine] = 2.0 * np.log(s)
    return F, U, logw.reshape(shape)


def _congruence(F: np.ndarray, U: np.ndarray, d: np.ndarray) -> np.ndarray:
    """``F U diag(d) U^T F^T``, symmetrised."""
    FU = F @ U
    return sym((FU * d[..., None, :]) @ transpose(FU))


# --------------- AIRM maps -------------------------------
def airm_inner(P: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Riemannian inner product ``tr(P^{-1} U P^{-1} V) = <F^{-1} U F^{-T}, F^{-1} V F^{-T}>_F``."""
    P, U, V = np.broadcast_arrays(np.asarray(P, float), np.asarray(U, float), np.asarray(V, float))
    F = spd_factor(sym(P))
    u, v = _whiten(F, U), _whiten(F, V)
    return np.einsum("...ij,...ij->...", u, v)


def airm_norm(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Riemannian norm ``||V||_P = sqrt(<V, V>_P)``."""
    return np.sqrt(np.maximum(airm_inner(P, V, V), 0.0))


def airm_exp(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    P, V = np.broadcast_arrays(np.asarray(P, float), np.asarray(V, float))
    F = spd_factor(sym(P))
    return sym(F @ expm_sym(_whiten(F, sym(V))) @ transpose(F))


def airm_log(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Logarithm map ``Log_P(Q) = P^{1/2} logm(P^{-1/2} Q P^{-1/2}) P^{1/2} = F logm(F^{-1} Q F^{-T}) F^T``."""
    F, U, logw = _whitened_eigh(P, Q)
    return _congruence(F, U, logw)


def airm_geodesic(P: np.ndarray, Q: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Point ``gamma(t) = P^{1/2} (P^{-1/2} Q P^{-1/2})^t P^{1/2}`` on the geodesic from P to Q."""
    F, U, logw = _whitened_eigh(P, Q)
    with np.errstate(over="ignore"):
        wt = np.exp(np.asarray(t, dtype=float)[..., None] * logw)
    _require_positive(wt, "airm_geodesic")
    return _congruence(F, U, wt)


def airm_transport(P: np.ndarray, Q: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Parallel transport of ``V`` from ``T_P`` to ``T_Q`` along the connecting geodesic."""
    F, U, logw = _whitened_eigh(P, Q)
    Mh = (U * np.exp(0.5 * logw)[..., None, :]) @ transpose(U)
    x = _whiten(F, np.broadcast_to(np.asarray(V, float), F.shape))
    FMh = F @ Mh
    return sym(FMh @ x @ transpose(FMh))


def airm_distance_sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared geodesic distance ``||logm(A^{-1/2} B A^{-1/2})||_F^2 = sum_i log^2 lambda_i(A^{-1} B)``."""
    return np.sum(_whitened_eigh(A, B, vectors=False)[2] ** 2, axis=-1)


def airm_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Geodesic distance under the affine-invariant metric."""
    return np.sqrt(airm_distance_sq(A, B))


def normalized_coordinates(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Transport ``V in T_P`` to the identity along the geodesic: ``P^{-1/2} V P^{-1/2}``."""
    Pih = sqrt_invsqrt_spd(P)[1]
    return sym(Pih @ V @ Pih)


# --------------------------- Log-Euclidean helpers --------------------------
def le_distance_sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Log-Euclidean distance ``||logm(A) - logm(B)||_F^2``."""
    D = logm_spd(A) - logm_spd(B)
    return np.sum(D * D, axis=(-2, -1))


def le_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``||logm(A) - logm(B)||_F``; equals :func:`airm_distance` when ``A`` and ``B`` commute."""
    return np.sqrt(le_distance_sq(A, B))


def le_mean(Y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Log-Euclidean Fréchet mean ``expm(sum_i w_i logm(Y_i))`` of a sample ``Y`` of shape (N, n, n)."""
    X = logm_spd(Y)
    w = np.full(len(Y), 1.0 / len(Y)) if weights is None else np.asarray(weights, float) / np.sum(weights)
    return expm_sym(np.tensordot(w, X, axes=1))



__all__ = [
    "airm_inner",
    "airm_norm",
    "airm_exp",
    "airm_log",
    "airm_geodesic",
    "airm_transport",
    "airm_distance",
    "airm_distance_sq",
    "normalized_coordinates",
    "le_distance",
    "le_distance_sq",
    "le_mean",
]
