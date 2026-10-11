from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field

import numpy as np

from .data import MAIN_SCENARIOS, seed_key, simulate_scenario, wishart_observations
from .geometry.spd import SPDViolationError, is_spd
from .metrics.evaluation import DEFAULT_FLOORS, forecast_errors, summarize_errors
from .models import MODEL_CLASSES, HoltWintersSpec

MODEL_ORDER = tuple(MODEL_CLASSES)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    spec: HoltWintersSpec
    noise_dof: int | None = None


CONFIGS = {
    "main": ExperimentConfig(
        "main", "Holt linear trend: alpha, beta estimated, phi = 1; noise-free observations", HoltWintersSpec()
    ),
    "ses": ExperimentConfig("ses", "Level only (beta = 0): simple exponential smoothing", HoltWintersSpec(trend=False)),
    "damped": ExperimentConfig("damped", "Damped trend: alpha, beta, phi estimated", HoltWintersSpec(damped=True)),
    "noisy": ExperimentConfig(
        "noisy",
        "Holt trend fitted to Wishart(nu = 50) realised covariances; scored against the latent truth",
        HoltWintersSpec(),
        noise_dof=50,
    ),
}


# Supplementary configurations. Common fitting criteria (experiment_protocol.md §10.1): the same
# specification as main / damped / noisy, with every model fitted under the AIRM or Frobenius distance.
EXTRA_CONFIGS = {
    f"{base}_fit{tag}": ExperimentConfig(
        f"{base}_fit{tag}",
        f"{CONFIGS[base].description}; all models fitted under the {metric} criterion",
        HoltWintersSpec(**{**_spec_kwargs, "fit_metric": metric}),
        noise_dof=CONFIGS[base].noise_dof,
    )
    for base, _spec_kwargs in (("main", {}), ("damped", {"damped": True}), ("noisy", {}))
    for tag, metric in (("airm", "airm"), ("frob", "frobenius"))
}
# Wishart degrees of freedom of the noisy configuration (sensitivity sweep, §10.7).
EXTRA_CONFIGS.update({
    f"noisy_nu{nu}": ExperimentConfig(
        f"noisy_nu{nu}", f"Holt trend fitted to Wishart(nu = {nu}) realised covariances; scored against the latent truth",
        HoltWintersSpec(), noise_dof=nu,
    )
    for nu in (10, 25, 50, 200)
})


def get_config(name: str) -> ExperimentConfig:
    return CONFIGS[name] if name in CONFIGS else EXTRA_CONFIGS[name]


@dataclass(frozen=True)
class StudyCell:
    scenario: str
    configs: tuple[str, ...]


MAIN_CONFIGS = tuple(CONFIGS)
FITLOSS_CONFIGS = ("main", "main_fitairm", "main_fitfrob", "damped", "damped_fitairm", "damped_fitfrob",
                   "noisy", "noisy_fitairm", "noisy_fitfrob")
SWEEP_CELLS = (
    # Scenario 1: drift-to-noise ratio v / sigma_omega (v = 0.05 fixed); 0.02 is the main study.
    *(StudyCell(f"rotation[sigma={s:g}]", ("main", "damped")) for s in (0.01, 0.02, 0.035, 0.05)),
    # Scenario 2: CIR mean-reversion speed and Feller ratio; (0.05, 1.1) is the main study.
    *(StudyCell(f"cir[kappa={k:g},feller={f:g}]", ("main", "noisy")) for k, f in ((0.05, 1.1), (0.05, 2.0), (0.05, 5.0), (0.02, 1.1), (0.2, 1.1))),
    # Wishart degrees of freedom on every main-study scenario; nu = 50 is the main study's noisy configuration.
    *(StudyCell(sc, (f"noisy_nu{nu}",)) for sc in MAIN_SCENARIOS for nu in (10, 25, 50, 200)),
)
STUDIES: dict[str, tuple[StudyCell, ...]] = {
    "main": tuple(StudyCell(sc, MAIN_CONFIGS) for sc in MAIN_SCENARIOS),
    "fitloss": (StudyCell("rotation", FITLOSS_CONFIGS),),
    "sweep": SWEEP_CELLS,
    "dim10": (StudyCell("rotation_n10", ("main", "damped", "noisy")),),
    "dcc": (StudyCell("dcc", ("main", "damped", "noisy")),),
}


@dataclass(frozen=True)
class RunSettings:
    n_steps: int = 500
    n_train: int = 350
    horizons: tuple[int, ...] = (1, 5, 10, 20)
    floors: tuple[float, ...] = DEFAULT_FLOORS
    base_seed: int = 20260924
    configs: tuple[str, ...] = tuple(CONFIGS)
    keep_forecasts_for_rep: int = 0
    kept_horizons: tuple[int, ...] = (1, 10)
    keep_data: bool = True  

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReplicationOutput:
    scenario: str
    rep: int
    rows: list[dict]
    data: dict[str, np.ndarray]
    forecasts: dict[str, np.ndarray] = field(default_factory=dict)
    seconds: float = 0.0


def replication_rng(base_seed: int, scenario: str, rep: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([base_seed, seed_key(scenario), rep, stream]))


def evaluate_model(
    model_name: str,
    config: ExperimentConfig,
    Y_obs: np.ndarray,
    Y_true: np.ndarray,
    settings: RunSettings,
) -> tuple[list[dict], dict[int, np.ndarray]]:
    
    cls = MODEL_CLASSES[model_name]
    base = {"config": config.name, "model": model_name}
    N, n_train = len(Y_obs), settings.n_train
    try:
        model = cls(config.spec).fit(Y_obs[:n_train])
        result = model.filter(Y_obs)
    except (SPDViolationError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return [{**base, "horizon": h, "status": "error", "error": repr(exc)} for h in settings.horizons], {}

    p = model.params_
    common = {
        **base,
        "status": "ok",
        "alpha": p.alpha,
        "beta": p.beta,
        "gamma": p.gamma,
        "phi": p.phi,
        "fit_loss": model.fit_result_.loss,
        "fit_seconds": model.fit_seconds_,
        "fit_evaluations": model.fit_result_.n_evaluations,
        "filter_us_per_step": 1e6 * result.elapsed / (N - result.start),
        "level_valid_ratio": float(np.mean(is_spd(result.level[result.start - 1 :]))),
    }
    rows, kept = [], {}
    for h in settings.horizons:
        origins = np.arange(n_train - 1, N - h)
        try:
            F = model.forecast(result, origins, [h])[:, 0]
        except (SPDViolationError, FloatingPointError, np.linalg.LinAlgError) as exc:

            rows.append({**common, "horizon": h, "status": "error", "error": repr(exc)})
            continue
        summary = summarize_errors(forecast_errors(F, Y_true[origins + h], settings.floors))
        rows.append({**common, "horizon": h, **summary})
        if h in settings.kept_horizons:
            kept[h] = F
    return rows, kept


def run_replication(scenario: str, rep: int, settings: RunSettings) -> ReplicationOutput:
    t0 = time.perf_counter()
    try:
        path = simulate_scenario(scenario, settings.n_steps, replication_rng(settings.base_seed, scenario, rep, 0))
        data = {"Sigma": path.Sigma, "eigenvalues": path.eigenvalues, "rotations": path.rotations}
        observations = {None: path.Sigma}
        noise_rng = replication_rng(settings.base_seed, scenario, rep, 1)
        for dof in sorted({get_config(c).noise_dof for c in settings.configs} - {None}):
            observations[dof] = wishart_observations(path.Sigma, dof, noise_rng)
            data[f"obs_nu{dof}"] = observations[dof]
    except Exception:
        error = "simulation failed: " + traceback.format_exc(limit=3)
        rows = [
            {"scenario": scenario, "rep": rep, "config": c, "model": m, "horizon": h, "status": "error", "error": error}
            for c in settings.configs for m in MODEL_ORDER for h in settings.horizons
        ]
        return ReplicationOutput(scenario, rep, rows, {}, {}, time.perf_counter() - t0)

    if not settings.keep_data:
        data = {}
    rows, forecasts = [], {}
    for config_name in settings.configs:
        config = get_config(config_name)
        for model_name in MODEL_ORDER:
            try:
                model_rows, kept = evaluate_model(model_name, config, observations[config.noise_dof], path.Sigma, settings)
            except Exception:  # never lose a whole replication silently: record and continue
                model_rows = [
                    {"config": config_name, "model": model_name, "horizon": h, "status": "error", "error": traceback.format_exc(limit=3)}
                    for h in settings.horizons
                ]
                kept = {}
            rows += [{"scenario": scenario, "rep": rep, **r} for r in model_rows]
            if rep == settings.keep_forecasts_for_rep and settings.keep_data:
                forecasts.update({f"{config_name}/{model_name}/h{h}": F for h, F in kept.items()})
    return ReplicationOutput(scenario, rep, rows, data, forecasts, time.perf_counter() - t0)


def benchmark_models(scenarios: tuple[str, ...], n_reps: int, settings: RunSettings, repeats: int = 3) -> list[dict]:
    rows = []
    spec = CONFIGS["main"].spec
    for scenario in scenarios:
        for rep in range(n_reps):
            try:
                Y = simulate_scenario(scenario, settings.n_steps, replication_rng(settings.base_seed, scenario, rep, 0)).Sigma
            except Exception:
                continue  # already recorded as a failed replication by the Monte Carlo run; nothing to time
            origins = np.arange(settings.n_train - 1, settings.n_steps - max(settings.horizons))
            for name in MODEL_ORDER:
                model = MODEL_CLASSES[name](spec).fit(Y[: settings.n_train])
                filter_seconds, forecast_seconds = np.inf, np.inf
                for _ in range(repeats):
                    result = model.filter(Y)
                    filter_seconds = min(filter_seconds, result.elapsed)
                    t0 = time.perf_counter()
                    model.forecast(result, origins, settings.horizons)
                    forecast_seconds = min(forecast_seconds, time.perf_counter() - t0)
                rows.append(
                    {
                        "scenario": scenario,
                        "rep": rep,
                        "model": name,
                        "fit_seconds": model.fit_seconds_,
                        "fit_evaluations": model.fit_result_.n_evaluations,
                        "filter_us_per_step": 1e6 * filter_seconds / (len(Y) - result.start),
                        "forecast_us_per_matrix": 1e6 * forecast_seconds / (len(origins) * len(settings.horizons)),
                    }
                )
    return rows
