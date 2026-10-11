from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np

from ..geometry.so3 import random_rotation
from .simulators import (
    cir_exact,
    compose_spd,
    dcc_garch,
    feller_ratio,
    log_ou_exact,
    merton_jump_diffusion,
    skew,
    so3_random_walk,
    son_random_walk,
)


@dataclass(frozen=True)
class RotationScenario:
    """Scenario 1 -- rapid rotation: fixed spectrum, eigenvectors follow a random walk on SO(3)."""

    name: str = "rotation"
    eigenvalues: tuple[float, float, float] = (4.0, 1.0, 0.25)
    sigma: float = 0.02
    drift_speed: float = 0.05


@dataclass(frozen=True)
class CIRScenario:
    """Scenario 2 -- eigenvalue extremes: static eigenvectors, independent CIR eigenvalues. """

    name: str = "cir"
    theta: tuple[float, float, float] = (2.0, 1.0, 0.5)
    kappa: float = 0.05
    feller: float = 1.1

    @property
    def sigma(self) -> np.ndarray:
        return np.sqrt(2.0 * self.kappa * np.asarray(self.theta) / self.feller)


@dataclass(frozen=True)
class JumpScenario:
    """Scenario 3 -- volatility spikes: Merton jump-diffusion eigenvalues."""

    name: str = "jump"
    base: tuple[float, float, float] = (2.0, 1.0, 0.5)
    mu: float = 0.0
    sigma: float = 0.05
    jump_intensity: float = 0.02
    jump_mean: float = 1.5
    jump_std: float = 0.5
    mean_reversion: float = 0.08
    rotation_sigma: float = 0.01


@dataclass(frozen=True)
class RotationOUScenario:
    """Scenario 1' (A') -- rotation with persistent drift *and* mean-reverting eigenvalues."""

    name: str = "rotation_ou"
    eigenvalues: tuple[float, float, float] = (4.0, 1.0, 0.25)
    sigma: float = 0.02
    drift_speed: float = 0.05
    kappa: float = 0.05
    log_sd: float = 0.25


@dataclass(frozen=True)
class RotationNScenario:
    """Scenario 1 in dimension n: fixed spectrum, eigenvectors follow a drifting random walk on SO(n)."""

    name: str = "rotation_n10"
    n: int = 10
    lambda_max: float = 4.0
    lambda_min: float = 0.25
    sigma: float = 0.02
    drift_speed: float = 0.05

    @property
    def eigenvalues(self) -> np.ndarray:
        return self.lambda_max * (self.lambda_min / self.lambda_max) ** (np.arange(self.n) / (self.n - 1))


@dataclass(frozen=True)
class DCCScenario:
    """Realistic covariance dynamics: DCC-GARCH(1,1) conditional covariances of n = 5 daily returns."""

    name: str = "dcc"
    variances: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0)
    garch_a: tuple[float, ...] = (0.05, 0.06, 0.07, 0.08, 0.09)
    persistence: float = 0.98
    dcc_a: float = 0.03
    dcc_b: float = 0.95
    rho: float = 0.4
    burn_in: int = 250


@dataclass
class ScenarioPath:
    name: str
    Sigma: np.ndarray
    eigenvalues: np.ndarray
    rotations: np.ndarray
    info: dict = field(default_factory=dict)


def simulate_rotation(n_steps: int, rng: np.random.Generator, cfg: RotationScenario = RotationScenario()) -> ScenarioPath:
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    R = so3_random_walk(n_steps, rng, cfg.sigma, drift=cfg.drift_speed * axis)
    lam = np.broadcast_to(np.asarray(cfg.eigenvalues, dtype=float), (n_steps, 3)).copy()
    return ScenarioPath(cfg.name, compose_spd(R, lam), lam, R, {"config": asdict(cfg), "axis": axis})


def simulate_cir(n_steps: int, rng: np.random.Generator, cfg: CIRScenario = CIRScenario()) -> ScenarioPath:
    theta = np.asarray(cfg.theta, dtype=float)
    kappa = np.full(3, cfg.kappa)
    lam = cir_exact(n_steps, rng, kappa, theta, cfg.sigma, x0=theta)
    R = np.broadcast_to(random_rotation(rng), (n_steps, 3, 3)).copy()
    info = {"config": asdict(cfg), "sigma": cfg.sigma, "feller_ratio": feller_ratio(kappa, theta, cfg.sigma)}
    return ScenarioPath(cfg.name, compose_spd(R, lam), lam, R, info)


def simulate_jump(n_steps: int, rng: np.random.Generator, cfg: JumpScenario = JumpScenario()) -> ScenarioPath:
    base = np.asarray(cfg.base, dtype=float)
    lam, jumps = merton_jump_diffusion(
        n_steps,
        rng,
        x0=base,
        mu=cfg.mu,
        sigma=cfg.sigma,
        jump_intensity=cfg.jump_intensity,
        jump_mean=cfg.jump_mean,
        jump_std=cfg.jump_std,
        mean_reversion=cfg.mean_reversion,
        long_run=base,
    )
    R = so3_random_walk(n_steps, rng, cfg.rotation_sigma)
    return ScenarioPath(cfg.name, compose_spd(R, lam), lam, R, {"config": asdict(cfg), "jumps": jumps})


def simulate_rotation_ou(n_steps: int, rng: np.random.Generator, cfg: RotationOUScenario = RotationOUScenario()) -> ScenarioPath:
    rot = simulate_rotation(n_steps, rng, RotationScenario(eigenvalues=cfg.eigenvalues, sigma=cfg.sigma, drift_speed=cfg.drift_speed))
    lam = log_ou_exact(n_steps, rng, np.log(np.asarray(cfg.eigenvalues, dtype=float)), cfg.kappa, cfg.log_sd)
    return ScenarioPath(cfg.name, compose_spd(rot.rotations, lam), lam, rot.rotations, {"config": asdict(cfg), "axis": rot.info["axis"]})


def simulate_rotation_n(n_steps: int, rng: np.random.Generator, cfg: RotationNScenario = RotationNScenario()) -> ScenarioPath:
    n, k = cfg.n, cfg.n * (cfg.n - 1) // 2
    U = skew(rng.standard_normal(k), n)
    U *= np.sqrt(2.0) / np.linalg.norm(U)
    sigma_n = cfg.sigma * np.sqrt(6.0 / (n * (n - 1)))
    R = son_random_walk(n_steps, rng, n, sigma_n, drift=cfg.drift_speed * U)
    lam = np.broadcast_to(cfg.eigenvalues, (n_steps, n)).copy()
    return ScenarioPath(cfg.name, compose_spd(R, lam), lam, R, {"config": asdict(cfg), "drift": cfg.drift_speed * U, "sigma_n": sigma_n})


def simulate_dcc(n_steps: int, rng: np.random.Generator, cfg: DCCScenario = DCCScenario()) -> ScenarioPath:
    n = len(cfg.variances)
    a = np.asarray(cfg.garch_a, dtype=float)
    corr = np.full((n, n), cfg.rho) + (1.0 - cfg.rho) * np.eye(n)
    H, _ = dcc_garch(n_steps, rng, cfg.variances, a, cfg.persistence - a, cfg.dcc_a, cfg.dcc_b, corr, cfg.burn_in)
    lam, V = np.linalg.eigh(H)
    return ScenarioPath(cfg.name, H, lam, V, {"config": asdict(cfg)})


SCENARIOS = {
    "rotation": simulate_rotation,
    "cir": simulate_cir,
    "jump": simulate_jump,
    "rotation_ou": simulate_rotation_ou,
    "rotation_n10": simulate_rotation_n,
    "dcc": simulate_dcc,
}
"""Base scenarios: name -> generator ``f(n_steps, rng, cfg)``."""

DEFAULT_CONFIGS = {
    "rotation": RotationScenario(),
    "cir": CIRScenario(),
    "jump": JumpScenario(),
    "rotation_ou": RotationOUScenario(),
    "rotation_n10": RotationNScenario(),
    "dcc": DCCScenario(),
}
SEED_KEYS = {"rotation": 0, "cir": 1, "jump": 2, "rotation_ou": 0, "rotation_n10": 4, "dcc": 5}
MAIN_SCENARIOS = ("rotation", "rotation_ou", "cir", "jump")
SCENARIO_TITLES = {
    "rotation": "S1 Rapid rotation (SO(3) random walk)",
    "rotation_ou": "S1' Rotation + log-OU eigenvalues",
    "cir": "S2 Eigenvalue extremes (CIR)",
    "jump": "S3 Volatility spikes (Merton jump-diffusion)",
    "rotation_n10": "S1 at n = 10 (SO(10) random walk)",
    "dcc": "DCC-GARCH (n = 5)",
}


def parse_scenario(name: str) -> tuple[str, dict[str, float]]:

    base, _, rest = name.partition("[")
    if base not in SCENARIOS:
        raise ValueError(f"unknown scenario {base!r}; choose from {sorted(SCENARIOS)}")
    overrides = {}
    if rest:
        if not rest.endswith("]"):
            raise ValueError(f"malformed scenario variant {name!r}")
        allowed = {f.name for f in fields(DEFAULT_CONFIGS[base])}
        for item in filter(None, rest[:-1].split(",")):
            key, _, value = item.partition("=")
            if key not in allowed or key == "name":
                raise ValueError(f"{base!r} has no parameter {key!r}")
            overrides[key] = type(getattr(DEFAULT_CONFIGS[base], key))(float(value))
    return base, overrides


def scenario_config(name: str):
    """Configuration of a scenario or variant."""
    base, overrides = parse_scenario(name)
    return replace(DEFAULT_CONFIGS[base], **overrides)


def seed_key(name: str) -> int:
    return SEED_KEYS[parse_scenario(name)[0]]


def simulate_scenario(name: str, n_steps: int, rng: np.random.Generator) -> ScenarioPath:
    """Simulate one path of the named scenario (or variant) with its configuration."""
    base, overrides = parse_scenario(name)
    if not overrides:
        return SCENARIOS[base](n_steps, rng)
    return SCENARIOS[base](n_steps, rng, scenario_config(name))
