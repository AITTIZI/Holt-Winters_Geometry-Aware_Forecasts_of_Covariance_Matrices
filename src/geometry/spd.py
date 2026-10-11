"""Linear-algebra primitives on the cone of symmetric positive-definite matrices."""

from __future__ import annotations

import numpy as np

_EPS = np.finfo(float).eps


class SPDViolationError(ValueError):
    """Raised when a matrix that must be symmetric positive definite is not."""


def transpose(A: np.ndarray) -> np.ndarray:
    return np.swapaxes(A, -1, -2)


def sym(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + transpose(A))


def eigh(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.linalg.eigh(sym(A))


def from_eig(w: np.ndarray, V: np.ndarray) -> np.ndarray:
    return sym((V * w[..., None, :]) @ transpose(V))


def positivity_threshold(w: np.ndarray) -> np.ndarray:
    n = w.shape[-1]
    return n * _EPS * np.max(np.abs(w), axis=-1)


def positive_mask(w: np.ndarray) -> np.ndarray:
    finite = np.all(np.isfinite(w), axis=-1)
    with np.errstate(invalid="ignore"):
        positive = np.all(w > positivity_threshold(w)[..., None], axis=-1)
    return finite & positive


def _require_positive(w: np.ndarray, what: str) -> None:
    ok = positive_mask(w)
    if not np.all(ok):
        bad = np.min(np.where(np.isfinite(w), w, -np.inf))
        raise SPDViolationError(
            f"{what}: argument is not numerically positive definite "
            f"(smallest eigenvalue {bad:.3e})"
        )


def expm_sym(X: np.ndarray) -> np.ndarray:
    w, V = eigh(X)
    with np.errstate(over="ignore"):
        ew = np.exp(w)
    _require_positive(ew, "expm_sym")
    return from_eig(ew, V)


def logm_spd(A: np.ndarray) -> np.ndarray:
    w, V = eigh(A)
    _require_positive(w, "logm_spd")
    return from_eig(np.log(w), V)


def powm_spd(A: np.ndarray, p: float | np.ndarray) -> np.ndarray:
    w, V = eigh(A)
    _require_positive(w, "powm_spd")
    p = np.asarray(p, dtype=float)[..., None]
    with np.errstate(over="ignore"):
        wp = np.exp(p * np.log(w))
    _require_positive(wp, "powm_spd")
    return from_eig(wp, V)


def sqrtm_spd(A: np.ndarray) -> np.ndarray:
    w, V = eigh(A)
    _require_positive(w, "sqrtm_spd")
    return from_eig(np.sqrt(w), V)


def invsqrtm_spd(A: np.ndarray) -> np.ndarray:
    w, V = eigh(A)
    _require_positive(w, "invsqrtm_spd")
    return from_eig(1.0 / np.sqrt(w), V)


def sqrt_invsqrt_spd(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, V = eigh(A)
    _require_positive(w, "sqrt_invsqrt_spd")
    s = np.sqrt(w)
    return from_eig(s, V), from_eig(1.0 / s, V)


def spectral_factor(A: np.ndarray) -> np.ndarray:
    w, V = eigh(A)
    if not np.all(w > 0):
        raise SPDViolationError(f"spectral_factor: argument is not positive definite (smallest eigenvalue {np.min(w):.3e})")
    return V * np.sqrt(w)[..., None, :]


def spd_factor(A: np.ndarray) -> np.ndarray:

    A = np.asarray(A, dtype=float)
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        pass

    flat = A.reshape(-1, *A.shape[-2:])
    F = np.empty_like(flat)
    for i, Ai in enumerate(flat):
        try:
            F[i] = np.linalg.cholesky(Ai)
        except np.linalg.LinAlgError:
            F[i] = spectral_factor(Ai)
    return F.reshape(A.shape)


def is_symmetric(A: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    scale = np.maximum(np.max(np.abs(A), axis=(-2, -1)), np.finfo(float).tiny)
    return np.max(np.abs(A - transpose(A)), axis=(-2, -1)) <= rtol * scale


def is_spd(A: np.ndarray, sym_rtol: float = 1e-10) -> np.ndarray:

    A = np.asarray(A, dtype=float)
    finite = np.all(np.isfinite(A), axis=(-2, -1))
    safe = np.where(finite[..., None, None], A, np.eye(A.shape[-1]))
    w = np.linalg.eigvalsh(sym(safe))
    return finite & is_symmetric(safe, sym_rtol) & positive_mask(w)


def assert_spd(A: np.ndarray, what: str = "matrix") -> None:
    ok = is_spd(A)
    if not np.all(ok):
        n_bad = int(np.size(ok) - np.count_nonzero(ok))
        raise SPDViolationError(f"{what}: {n_bad} of {np.size(ok)} matrices are not SPD")


REPAIR_FLOOR = 1e-3



def repair_spd(A: np.ndarray, floor_rel: float) -> np.ndarray:
    w, V = eigh(A)
    floor = floor_rel * np.max(np.abs(w), axis=-1, keepdims=True)
    return from_eig(np.maximum(w, floor), V)


def random_spd(n: int, rng: np.random.Generator, size: tuple[int, ...] = (), log_scale: float = 1.0) -> np.ndarray:
    G = rng.standard_normal(size + (n, n))
    return expm_sym(log_scale * sym(G))
