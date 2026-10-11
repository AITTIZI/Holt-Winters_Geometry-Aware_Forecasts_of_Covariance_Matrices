"""The rotation group SO(3) and its Lie algebra so(3) of skew-symmetric matrices (Haar sampling for any SO(n))."""

from __future__ import annotations

import numpy as np


def hat(omega: np.ndarray) -> np.ndarray:
    omega = np.asarray(omega, dtype=float)
    x, y, z = omega[..., 0], omega[..., 1], omega[..., 2]
    O = np.zeros(omega.shape[:-1] + (3, 3))
    O[..., 0, 1], O[..., 0, 2] = -z, y
    O[..., 1, 0], O[..., 1, 2] = z, -x
    O[..., 2, 0], O[..., 2, 1] = -y, x
    return O


def vee(S: np.ndarray) -> np.ndarray:
    S = 0.5 * (S - np.swapaxes(S, -1, -2))
    return np.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], axis=-1)


def expm_so3(omega: np.ndarray) -> np.ndarray:
    omega = np.asarray(omega, dtype=float)
    theta = np.linalg.norm(omega, axis=-1)[..., None, None]
    W = hat(omega)
    W2 = W @ W
    small = theta < 1e-8
    safe = np.where(small, 1.0, theta)
    a = np.where(small, 1.0 - theta**2 / 6.0, np.sin(safe) / safe)
    b = np.where(small, 0.5 - theta**2 / 24.0, (1.0 - np.cos(safe)) / safe**2)
    return np.eye(3) + a * W + b * W2


def random_rotation(rng: np.random.Generator, n: int = 3) -> np.ndarray:
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def orthogonality_error(R: np.ndarray) -> np.ndarray:
    I = np.eye(R.shape[-1])
    return np.max(np.abs(np.swapaxes(R, -1, -2) @ R - I), axis=(-2, -1))


def is_rotation(R: np.ndarray, atol: float = 1e-10) -> np.ndarray:
    return (orthogonality_error(R) <= atol) & (np.abs(np.linalg.det(R) - 1.0) <= atol)
