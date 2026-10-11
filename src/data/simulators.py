from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from ..geometry.so3 import expm_so3, random_rotation
from ..geometry.spd import SPDViolationError, sym


def so3_random_walk(
    n_steps: int,
    rng: np.random.Generator,
    sigma: float,
    drift: np.ndarray | None = None,
    R0: np.ndarray | None = None,
) -> np.ndarray:
    """Random walk on SO(3): ``R_t = R_{t-1} exp(S_t)``, ``S_t = hat(drift + sigma * eps_t)``.
        ``eps_t ~ N(0, I_3)``
    """
    drift = np.zeros(3) if drift is None else np.asarray(drift, dtype=float)
    R0 = random_rotation(rng) if R0 is None else np.asarray(R0, dtype=float)
    increments = expm_so3(drift + sigma * rng.standard_normal((n_steps - 1, 3)))
    R = np.empty((n_steps, 3, 3))
    R[0] = R0
    for t in range(1, n_steps):
        R[t] = R[t - 1] @ increments[t - 1]
    return R


def skew(x: np.ndarray, n: int) -> np.ndarray:
    """Skew-symmetric (..., n, n) matrices whose strictly upper triangles hold ``-x`` (row-major), ``x`` of shape (..., n(n-1)/2)."""
    x = np.asarray(x, dtype=float)
    iu = np.triu_indices(n, 1)
    A = np.zeros(x.shape[:-1] + (n, n))
    A[..., iu[0], iu[1]] = -x
    return A - np.swapaxes(A, -1, -2)


def son_random_walk(
    n_steps: int,
    rng: np.random.Generator,
    n: int,
    sigma: float,
    drift: np.ndarray | None = None,
    R0: np.ndarray | None = None,
) -> np.ndarray:
    """Random walk on SO(n): ``R_t = R_{t-1} expm(D + sigma * skew(eps_t))``, ``eps_t ~ N(0, I_{n(n-1)/2})``.
    ``D`` is a fixed skew-symmetric drift (body-frame angular velocity).
    """
    D = np.zeros((n, n)) if drift is None else np.asarray(drift, dtype=float)
    R0 = random_rotation(rng, n) if R0 is None else np.asarray(R0, dtype=float)
    noise = skew(rng.standard_normal((n_steps - 1, n * (n - 1) // 2)), n)
    R = np.empty((n_steps, n, n))
    R[0] = R0
    for t in range(1, n_steps):
        R[t] = R[t - 1] @ expm(D + sigma * noise[t - 1])
    return R


def log_ou_exact(
    n_steps: int,
    rng: np.random.Generator,
    mean_log: np.ndarray,
    kappa: float,
    stationary_sd: float,
    u0: np.ndarray | None = None,
) -> np.ndarray:
    """Independent exact log-Ornstein–Uhlenbeck processes ``du = kappa (m - u) dt + s sqrt(2 kappa) dW``, ``x = e^u``.

    Exact Gaussian AR(1) transition with ``dt = 1``:
    ``u_{t+1} = m + (u_t - m) e^{-kappa} + s sqrt(1 - e^{-2 kappa}) eps``, so the stationary law of
    ``u`` is N(m, s^2) and there is no discretisation bias. Returns ``x = exp(u)``, (n_steps, k), > 0.
    """
    m = np.atleast_1d(np.asarray(mean_log, dtype=float))
    decay = np.exp(-kappa)
    scale = stationary_sd * np.sqrt(1.0 - decay**2)
    u = np.empty((n_steps, len(m)))
    u[0] = m if u0 is None else u0
    eps = rng.standard_normal((n_steps - 1, len(m)))
    for t in range(1, n_steps):
        u[t] = m + (u[t - 1] - m) * decay + scale * eps[t - 1]
    return np.exp(u)


def dcc_garch(
    n_steps: int,
    rng: np.random.Generator,
    variances: np.ndarray,
    garch_a: np.ndarray,
    garch_b: np.ndarray,
    dcc_a: float,
    dcc_b: float,
    correlation: np.ndarray,
    burn_in: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """DCC-GARCH(1,1) conditional covariances ``H_t = D_t R_t D_t`` """
    v, a, b = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (variances, garch_a, garch_b))
    Qbar = np.asarray(correlation, dtype=float)
    n = len(v)
    if np.any(a < 0) or np.any(b < 0) or np.any(a + b >= 1) or dcc_a < 0 or dcc_b < 0 or dcc_a + dcc_b >= 1:
        raise ValueError("DCC-GARCH parameters violate covariance stationarity")
    if not np.allclose(np.diag(Qbar), 1.0) or np.linalg.eigvalsh(Qbar)[0] <= 0:
        raise ValueError("Qbar must be a positive definite correlation matrix")
    omega = v * (1.0 - a - b)
    total = n_steps + burn_in
    H = np.empty((total, n, n))
    r = np.empty((total, n))
    h, Q = v.copy(), Qbar.copy()
    z = rng.standard_normal((total, n))
    for t in range(total):
        if t > 0:
            e = r[t - 1] / np.sqrt(h_prev)
            h = omega + a * r[t - 1] ** 2 + b * h_prev
            Q = (1.0 - dcc_a - dcc_b) * Qbar + dcc_a * np.outer(e, e) + dcc_b * Q
        q = 1.0 / np.sqrt(np.diag(Q))
        Rt = Q * q[:, None] * q[None, :]
        d = np.sqrt(h)
        H[t] = sym(Rt * d[:, None] * d[None, :])
        r[t] = np.linalg.cholesky(H[t]) @ z[t]
        h_prev = h
    return H[burn_in:], r[burn_in:]


def feller_ratio(kappa: np.ndarray, theta: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """``2 kappa theta / sigma^2``; the CIR process never hits zero iff this is >= 1."""
    return 2.0 * np.asarray(kappa) * np.asarray(theta) / np.asarray(sigma) ** 2


def cir_exact(
    n_steps: int,
    rng: np.random.Generator,
    kappa: np.ndarray,
    theta: np.ndarray,
    sigma: np.ndarray,
    x0: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """Exact simulation of independent CIR processes ``dx = kappa (theta - x) dt + sigma sqrt(x) dW``.
        x_{t+dt} = c * chi'^2_d(x_t e^{-kappa dt} / c),
        c = sigma^2 (1 - e^{-kappa dt}) / (4 kappa),   d = 4 kappa theta / sigma^2.
    """
    kappa, theta, sigma, x0 = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (kappa, theta, sigma, x0))
    if np.any(feller_ratio(kappa, theta, sigma) < 1.0):
        raise ValueError("Feller condition 2*kappa*theta >= sigma^2 violated: zero would be attainable")
    decay = np.exp(-kappa * dt)
    c = sigma**2 * (1.0 - decay) / (4.0 * kappa)
    df = 4.0 * kappa * theta / sigma**2
    x = np.empty((n_steps, len(x0)))
    x[0] = x0
    for t in range(1, n_steps):
        x[t] = c * rng.noncentral_chisquare(df, x[t - 1] * decay / c)
        if not np.all(np.isfinite(x[t]) & (x[t] > 0)):
            raise SPDViolationError(f"CIR draw at step {t} is not strictly positive ({x[t].min():.3e})")
    return x


def merton_jump_diffusion(
    n_steps: int,
    rng: np.random.Generator,
    x0: np.ndarray,
    mu: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    mean_reversion: float = 0.0,
    long_run: np.ndarray | None = None,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Merton jump-diffusion (GBM + compound-Poisson log-normal jumps), optionally mean-reverting.

    With ``u_t = log x_t`` and ``N_t ~ Poisson(jump_intensity dt)``,

        u_{t+dt} = m + (u_t - m) e^{-kappa dt} + (mu - sigma^2/2) dt + sigma sqrt(dt) eps_t
                   + sum_{k=1}^{N_t} J_k,        J_k ~ N(jump_mean, jump_std^2),

    where ``m = log(long_run)`` and ``kappa = mean_reversion``. For ``kappa = 0``
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    m = np.log(x0 if long_run is None else np.asarray(long_run, dtype=float))
    k = len(x0)
    decay = np.exp(-mean_reversion * dt)
    u = np.empty((n_steps, k))
    jumps = np.zeros((n_steps, k), dtype=int)
    u[0] = np.log(x0)
    for t in range(1, n_steps):
        n_jumps = rng.poisson(jump_intensity * dt, size=k)
        jump = n_jumps * jump_mean + np.sqrt(n_jumps) * jump_std * rng.standard_normal(k)
        diffusion = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(k)
        u[t] = m + (u[t - 1] - m) * decay + diffusion + jump
        jumps[t] = n_jumps
    return np.exp(u), jumps


def compose_spd(R: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
    """``R diag(lambda) R^T`` for stacked rotations/eigenvalues (broadcasting)."""
    return sym((R * eigenvalues[..., None, :]) @ np.swapaxes(R, -1, -2))


def wishart_observations(Sigma: np.ndarray, dof: int, rng: np.random.Generator) -> np.ndarray:
    """Realised covariances ``C_t = (1/nu) sum_{k=1}^{nu} r_k r_k^T`` with ``r_k ~ N(0, Sigma_t)``."""
    N, n, _ = Sigma.shape
    if dof < n:
        raise ValueError("Wishart degrees of freedom must be >= dimension for an SPD sample")
    chol = np.linalg.cholesky(Sigma)
    returns = rng.standard_normal((N, dof, n)) @ np.swapaxes(chol, -1, -2)
    return sym(np.swapaxes(returns, -1, -2) @ returns / dof)
