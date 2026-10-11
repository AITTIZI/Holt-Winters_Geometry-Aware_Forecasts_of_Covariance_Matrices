from __future__ import annotations
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter  # noqa: E402

from ..data.scenarios import SCENARIO_TITLES  # noqa: E402
from ..geometry.airm import airm_distance  # noqa: E402
from ..geometry.spd import is_spd  # noqa: E402
from ..metrics.evaluation import floor_tag  # noqa: E402
from ..metrics.summary import PRIMARY_FLOOR, PRIMARY_METRIC  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Categorical slots 1-3 of the validated reference palette (all-pairs safe for three series); colour
# follows the model, so Euclidean and AIRM keep the slots they had before Log-Euclidean HW was added.
MODEL_STYLE = {
    "euclidean": {"color": "#2a78d6", "marker": "o", "label": "Euclidean HW"},
    "le": {"color": "#eb6834", "marker": "s", "label": "Log-Euclidean HW"},
    "airm": {"color": "#1baf7a", "marker": "^", "label": "AIRM HW"},
}
# Reserved status colour for invalid (non-SPD) forecasts: always paired with an "x" marker and a legend label.
NOT_SPD, NOT_SPD_FILL = "#8f1d1d", "#f6e1e1"
EIGEN_SHADES = (INK, INK2, MUTED)  # truth eigenvalues, largest -> smallest
CONFIG_LABELS = {"main": "Holt", "ses": "SES", "damped": "Damped", "noisy": "Holt, noisy obs."}
SHORT = {"rotation": "S1 rotation", "rotation_ou": "S1' rotation + OU", "cir": "S2 CIR", "jump": "S3 jumps"}

_REP = floor_tag(PRIMARY_FLOOR)
METRIC_LABELS = {
    f"airm_{_REP}_rmse": r"AIRM RMSE, $d_A(\hat\Sigma,\Sigma)$",
    f"le_{_REP}_rmse": r"LE RMSE, $d_{LE}(\hat\Sigma,\Sigma)$",
    "frob_rmse": r"Frobenius RMSE, $\|\hat\Sigma-\Sigma\|_F$",
    "airm_valid_rmse": r"AIRM RMSE (SPD forecasts only)",
}
REPAIR_NOTE = (
    "AIRM errors of non-SPD Euclidean forecasts are scored after eigenvalue repair at "
    rf"$10^{{{int(round(np.log10(PRIMARY_FLOOR)))}}}\,\lambda_{{\max}}$ (scoring only; the forecasts themselves are never repaired)."
)


def apply_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "legend.frameon": False,
            "legend.labelcolor": INK2,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _model_legend(
    fig: plt.Figure, extra: list | None = None, ncol: int = 4, y: float = 1.0, after: list | None = None, styles: dict | None = None
) -> None:
    handles = [plt.Line2D([], [], color=s["color"], marker=s["marker"], label=s["label"]) for s in (styles or MODEL_STYLE).values()]
    fig.legend(handles=(extra or []) + handles + (after or []), loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y))


def _not_spd_handles() -> list:
    return [
        plt.Line2D([], [], color=NOT_SPD, marker="x", linestyle="none", markersize=6, label="non-SPD forecast"),
        Patch(facecolor=NOT_SPD_FILL, edgecolor="none", label=r"outside the SPD cone ($\leq 0$)"),
    ]


def _footnote(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, -0.01, text, ha="left", va="top", fontsize=7.5, color=INK2, wrap=True)


def _mark_split(ax: plt.Axes, n_train: int) -> None:
    ax.axvline(n_train - 0.5, color=MUTED, linewidth=0.8)


def _ordered(scenarios) -> list[str]:

    present = set(scenarios)
    return [s for s in SHORT if s in present]


def _models(present) -> dict[str, dict]:
    present = set(present)
    return {m: st for m, st in MODEL_STYLE.items() if m in present}


def _plain_log_axis(ax: plt.Axes) -> None:
    lo, hi = ax.get_ylim()
    decades = np.log10(hi / lo) if lo > 0 else np.inf
    if decades < 1:
        ax.yaxis.set_major_locator(MaxNLocator(5))
    else:
        ax.yaxis.set_major_locator(LogLocator(10, subs=(1.0, 2.0, 5.0) if decades < 3 else (1.0,)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _signed_log_scale(ax: plt.Axes, values: np.ndarray) -> None:
    v = values[np.isfinite(values)]
    if np.all(v > 0):
        ax.set_yscale("log")
        return
    small = np.abs(v[v != 0])
    ax.set_yscale("symlog", linthresh=10.0 ** np.floor(np.log10(np.percentile(small, 5))) if small.size else 1.0)


def _shade_not_spd(ax: plt.Axes) -> None:
    """Shade the half-plane ``y <= 0`` (outside the SPD cone) if the axis reaches into it."""
    if ax.get_yscale() == "log":  # every plotted value is > 0 and zero is at -infinity
        return
    lo, hi = ax.get_ylim()
    if lo < 0:
        ax.axhspan(lo, 0.0, facecolor=NOT_SPD_FILL, edgecolor="none", zorder=0)
        ax.set_ylim(lo, hi)
    ax.axhline(0.0, color=INK2, linewidth=0.9, zorder=1)


# ----------------------------------------------------------------- data views
def plot_scenario_overview(paths: dict[str, np.ndarray], n_train: int, out: Path, colors: list[str] | None = None) -> Path:
    """Sorted eigenvalues, AIRM speed and condition number of one path per scenario."""
    paths = {s: paths[s] for s in _ordered(paths)}
    eig_c, speed_c, cond_c = (colors[:3], colors[3], colors[4]) if colors else (EIGEN_SHADES, INK, INK)
    fig, axes = plt.subplots(len(paths), 3, figsize=(11, 2.4 * len(paths)), sharex=True, squeeze=False)
    for row, (scenario, Sigma) in zip(axes, paths.items()):
        t = np.arange(len(Sigma))
        lam = np.linalg.eigvalsh(Sigma)[:, ::-1]
        for k in range(3):
            row[0].plot(t, lam[:, k], color=eig_c[k], linewidth=1.0)
        row[0].set_yscale("log")
        _plain_log_axis(row[0])
        row[0].set_ylabel(SHORT[scenario], color=INK, fontweight="bold")
        row[1].plot(t[1:], airm_distance(Sigma[:-1], Sigma[1:]), color=speed_c, linewidth=0.8)
        cond = lam[:, 0] / lam[:, 2]
        row[2].plot(t, cond, color=cond_c, linewidth=0.8)
        if cond.max() / cond.min() > 1.05:
            row[2].set_yscale("log")
            _plain_log_axis(row[2])
        else:  # isospectral path: constant condition number
            row[2].set_ylim(0.0, 2.0 * cond.max())
        for ax in row:
            _mark_split(ax, n_train)
    axes[0, 0].set_title(r"Eigenvalues $\lambda_1 \geq \lambda_2 \geq \lambda_3 > 0$ (log)")
    axes[0, 1].set_title(r"AIRM step length $d_A(\Sigma_{t-1}, \Sigma_t)$")
    axes[0, 2].set_title(r"Condition number $\lambda_1 / \lambda_3$")
    fig.legend(
        handles=[plt.Line2D([], [], color=eig_c[k], linewidth=1.0, label=rf"$\lambda_{k + 1}$") for k in range(3)],
        loc="upper left", ncol=3, bbox_to_anchor=(0.05, 1.04),
    )
    for ax in axes[-1]:
        ax.set_xlabel("time step (grey rule: train | test split)")
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout()
    return _save(fig, out)


def plot_forecast_properties(
    truth: dict[str, np.ndarray], forecasts: dict[str, dict[str, np.ndarray]], n_train: int, horizon: int, out: Path, config: str = "main"
) -> Path:
    """Trace, determinant and smallest eigenvalue of h-step forecasts against the truth (test window)."""
    truth = {s: truth[s] for s in _ordered(truth)}
    fig, axes = plt.subplots(len(truth), 3, figsize=(11, 2.5 * len(truth)), squeeze=False)
    props = {
        "trace": lambda A: np.trace(A, axis1=-2, axis2=-1),
        "det": np.linalg.det,
        "min_eig": lambda A: np.linalg.eigvalsh(A)[..., 0],
    }
    styles = _models(next(iter(forecasts.values())))
    for row, (scenario, Sigma) in zip(axes, truth.items()):
        times = np.arange(n_train - 1 + horizon, len(Sigma))
        valid = {model: is_spd(forecasts[scenario][model]) for model in styles}
        for ax, (prop, f) in zip(row, props.items()):
            ax.plot(times, f(Sigma[times]), color=INK, linewidth=1.8, zorder=5)
            values = [f(Sigma[times])]
            for model, style in styles.items():
                y = f(forecasts[scenario][model])
                values.append(y)
                ax.plot(times, y, color=style["color"], linewidth=1.1, marker=style["marker"], markevery=15, markersize=4)
                if prop == "min_eig" and not valid[model].all():
                    bad = ~valid[model]
                    ax.scatter(times[bad], y[bad], marker="x", color=NOT_SPD, s=16, linewidths=1.0, zorder=6)
            if prop == "det":
                _signed_log_scale(ax, np.concatenate(values))
            if prop in ("det", "min_eig"):
                _shade_not_spd(ax)  # det <= 0 or lambda_min <= 0 is outside the SPD cone
        row[0].set_ylabel(SHORT[scenario], color=INK, fontweight="bold")
    axes[0, 0].set_title(r"Trace $\mathrm{tr}\,\hat\Sigma$")
    axes[0, 1].set_title(r"Determinant $\det\hat\Sigma$ (log / symlog)")
    axes[0, 2].set_title(r"Smallest eigenvalue $\lambda_{\min}(\hat\Sigma)$")
    for ax in axes[-1]:
        ax.set_xlabel("target time step")
    _model_legend(fig, [plt.Line2D([], [], color=INK, linewidth=1.8, label="Truth")], ncol=6, after=_not_spd_handles(), styles=styles)
    fig.suptitle(
        f"{horizon}-step-ahead forecasts over the test window (replication 0, {CONFIG_LABELS.get(config, config)} configuration)",
        y=1.04, color=INK, fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out)


def plot_error_vs_horizon(summary: pd.DataFrame, configs: list[str], out: Path, colors: list[str] | None = None,
                          metric_in_ylabel: bool = True) -> Path:
    """Mean RMSE (primary metric) with 95% bootstrap CI against horizon; rows = configurations, cols = scenarios."""
    metric = summary.attrs.get("metric", PRIMARY_METRIC)
    present = _models(summary["model"])
    styles = {m: {**st, "color": c} for (m, st), c in zip(present.items(), colors)} if colors else present
    scenarios = _ordered(summary["scenario"])
    fig, axes = plt.subplots(len(configs), len(scenarios), figsize=(3.7 * len(scenarios), 2.5 * len(configs)), squeeze=False, sharex=True)
    for i, config in enumerate(configs):
        for j, scenario in enumerate(scenarios):
            ax = axes[i, j]
            g = summary[(summary["config"] == config) & (summary["scenario"] == scenario)]
            for model, style in styles.items():
                m = g[g["model"] == model].sort_values("horizon")
                ax.fill_between(m["horizon"], m["ci_low"], m["ci_high"], color=style["color"], alpha=0.15, linewidth=0)
                ax.plot(m["horizon"], m["mean"], color=style["color"], marker=style["marker"])
            ax.set_yscale("log")
            _plain_log_axis(ax)
            if i == 0:
                ax.set_title(SCENARIO_TITLES[scenario], fontsize=9)
            if j == 0:
                label = CONFIG_LABELS.get(config, config)
                ax.set_ylabel(f"{label}\nmean {METRIC_LABELS.get(metric, metric)}" if metric_in_ylabel else label)
            if i == len(configs) - 1:
                ax.set_xlabel("forecast horizon h")
    horizons = sorted(summary["horizon"].unique())
    for ax in axes[-1]:
        ax.set_xticks(horizons)
    _model_legend(fig, ncol=len(styles), styles=styles)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _footnote(fig, "Lines: mean over replications; bands: 95% percentile bootstrap CI of the mean. " + REPAIR_NOTE)
    return _save(fig, out)


