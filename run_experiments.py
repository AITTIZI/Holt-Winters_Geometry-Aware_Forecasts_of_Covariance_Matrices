from __future__ import annotations
import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse 
import dataclasses  
import json 
import logging  
import multiprocessing as mp 
import platform  
import sys  
import time 
from datetime import datetime, timezone  
from pathlib import Path  

import matplotlib 
import numpy as np  
import pandas as pd  
import scipy 

from src.data import DEFAULT_CONFIGS, MAIN_SCENARIOS 
from src.experiment import (  
    CONFIGS,
    MODEL_ORDER,
    STUDIES,
    RunSettings,
    StudyCell,
    benchmark_models,
    get_config,
    run_replication,
)
from src.geometry import airm_distance  
from src.metrics.summary import (  
    COMPARISONS,
    PRIMARY_METRIC,
    floor_sensitivity,
    markdown_table,
    pairwise_tests,
    params_summary,
    summary_table,
    volume_bias,
)
from src.models import MODEL_CLASSES  

log = logging.getLogger("spd-hw")
GEOMETRIC = tuple(name for name, cls in MODEL_CLASSES.items() if cls.preserves_spd)
BENCH_REPS = 5 
BENCH_SCENARIOS = (*MAIN_SCENARIOS, "rotation_n10", "dcc")
STUDY_FILES = {"main": "metrics.csv"}  
REAL = "realdata" 
RAW_DIR = Path("data") / "raw"


def metrics_file(study: str) -> str:
    return STUDY_FILES.get(study, f"{study}_metrics.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--studies", nargs="+", default=[*STUDIES, REAL], choices=[*STUDIES, REAL])
    p.add_argument("--reps", type=int, default=None, help="Monte Carlo replications per cell (default 100; 4 with --quick)")
    p.add_argument("--scenarios", nargs="+", default=None, choices=MAIN_SCENARIOS, help="main study: scenarios to run")
    p.add_argument("--configs", nargs="+", default=None, choices=list(CONFIGS), help="main study: configurations to run")
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--n-train", type=int, default=350)
    p.add_argument("--seed", type=int, default=RunSettings.base_seed)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--quick", action="store_true", help="4 replications; main study only, main configuration (unless overridden)")
    p.add_argument("--no-diagnostics", action="store_true", help="skip the noise-bias diagnostic re-fit")
    p.add_argument("--plots-only", action="store_true", help="re-run the analysis (summaries) on saved results, without re-fitting")
    p.add_argument("--benchmark-only", action="store_true", help="re-measure the single-process cost benchmark (idle machine), then re-run the analysis")
    args = p.parse_args()
    if args.quick:
        args.reps = 4 if args.reps is None else args.reps
        args.configs = args.configs or ["main"]
        if "--studies" not in sys.argv:
            args.studies = ["main"]
    args.reps = 100 if args.reps is None else args.reps
    args.scenarios = args.scenarios or list(MAIN_SCENARIOS)
    args.configs = args.configs or list(CONFIGS)
    return args


def setup_logging(out: Path) -> Path:
    (out / "logs").mkdir(parents=True, exist_ok=True)
    path = out / "logs" / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    return path


def write_realdata_features(summary_dir: Path) -> None:
    from src.data.realdata import DATASETS, load_monthly
    from src.realdata_experiment import series_features

    rows = []
    for name in DATASETS:
        if (RAW_DIR / DATASETS[name]).exists():
            series = load_monthly(name, RAW_DIR)
            rows.append(series_features(name, series.C, series.n_days, series.months))
    if rows:
        pd.DataFrame(rows).to_csv(summary_dir / "realdata_features.csv", index=False)


def run_realdata(out: Path, workers: int) -> dict:
    from src.data.realdata import DATASETS, FRENCH_URL, load_monthly, sha256
    from src.metrics.summary import COMPARISONS
    from src.realdata_experiment import audit_spd, run_rolling, series_features, summarise

    t0 = time.perf_counter()
    data, features, provenance = {}, [], []
    for name in DATASETS:
        existed = (RAW_DIR / DATASETS[name]).exists()
        series = load_monthly(name, RAW_DIR)
        path = RAW_DIR / DATASETS[name]
        provenance.append({"dataset": name, "file": DATASETS[name], "url": FRENCH_URL + DATASETS[name], "sha256": sha256(path),
                           "downloaded_now": not existed, "file_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
                           "source_note": series.source_note, "industries": " ".join(series.industries)})
        data[name] = series.C
        features.append(series_features(name, series.C, series.n_days, series.months))
    log.info("real data: %s", ", ".join(f"{k}: {v.shape}" for k, v in data.items()))
    per_origin = run_rolling(data, workers)
    (out / "realdata").mkdir(parents=True, exist_ok=True)
    per_origin.to_csv(out / "realdata" / "per_origin.csv.gz", index=False)
    summary, tests = summarise(per_origin, COMPARISONS)
    (out / "summary").mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary" / "realdata_summary.csv", index=False)
    tests.to_csv(out / "summary" / "realdata_tests.csv", index=False)
    pd.DataFrame(features).to_csv(out / "summary" / "realdata_features.csv", index=False)
    pd.DataFrame(provenance).to_csv(out / "summary" / "realdata_provenance.csv", index=False)
    violations = audit_spd(per_origin)
    if violations:
        log.error("INVARIANT VIOLATION: %d non-SPD LE/AIRM forecasts on the real data", violations)
    else:
        log.info("real data: SPD audit passed (%d LE/AIRM forecasts)", int(per_origin["model"].isin(GEOMETRIC).sum()))
    return {"seconds": time.perf_counter() - t0, "violations": violations}


def study_cells(study: str, args: argparse.Namespace) -> tuple[StudyCell, ...]:
    if study != "main":
        return STUDIES[study]
    configs = tuple(c for c in CONFIGS if c in args.configs)
    return tuple(StudyCell(c.scenario, configs) for c in STUDIES["main"] if c.scenario in args.scenarios)


def _task(task: tuple) -> tuple[str, object]:
    study, scenario, rep, settings = task
    return study, run_replication(scenario, rep, settings)


# ------------------------------- simulation --------------------------------
def run_monte_carlo(args: argparse.Namespace, base: RunSettings) -> None:
    out = args.out
    tasks = []
    for study in args.studies:
        for cell in study_cells(study, args):
            settings = dataclasses.replace(base, configs=cell.configs, keep_data=(study == "main"))
            tasks += [(study, cell.scenario, rep, settings) for rep in range(args.reps)]
    log.info("running %d replications (studies %s) on %d workers", len(tasks), ", ".join(args.studies), args.workers)
    rows = {s: [] for s in args.studies}
    data, forecasts = {}, {}
    t0 = time.perf_counter()
    # Longest tasks first (n = 10 replications), so the pool does not finish on a straggler.
    tasks.sort(key=lambda t: t[1] != "rotation_n10")
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (study, res) in enumerate(pool.imap_unordered(_task, tasks, chunksize=1), 1):
            rows[study] += [{"study": study, **r} for r in res.rows]
            if res.data: 
                data.setdefault(res.scenario, {})[res.rep] = res.data
            if res.forecasts:
                forecasts[res.scenario] = res.forecasts
            if i % 100 == 0 or i == len(tasks):
                log.info("  %d/%d replications done (%.0fs elapsed)", i, len(tasks), time.perf_counter() - t0)
    wall = time.perf_counter() - t0

    (out / "metrics").mkdir(parents=True, exist_ok=True)
    for study, study_rows in rows.items():
        df = pd.DataFrame(study_rows).sort_values(["config", "scenario", "rep", "model", "horizon"]).reset_index(drop=True)
        df.to_csv(out / "metrics" / metrics_file(study), index=False)
        log.info("study %s: %d metric rows saved", study, len(df))
    if data:
        (out / "data").mkdir(parents=True, exist_ok=True)
        for scenario, reps in data.items():
            order = sorted(reps)
            np.savez_compressed(out / "data" / f"{scenario}.npz", reps=np.array(order), **{k: np.stack([reps[r][k] for r in order]) for k in reps[order[0]]})
    if forecasts:
        (out / "forecasts").mkdir(parents=True, exist_ok=True)
        for scenario, kept in forecasts.items():
            np.savez_compressed(out / "forecasts" / f"{scenario}_rep{base.keep_forecasts_for_rep}.npz", **kept)

    log.info("benchmarking computational overhead in a single process ...")
    t1 = time.perf_counter()
    bench_scenarios = tuple(s for s in BENCH_SCENARIOS if s in args.scenarios or s not in MAIN_SCENARIOS)
    bench = pd.DataFrame(benchmark_models(bench_scenarios, BENCH_REPS if not args.quick else 1, base))
    bench.to_csv(out / "metrics" / "compute_benchmark.csv", index=False)
    bench_seconds = time.perf_counter() - t1

    diag_seconds = None
    if not args.no_diagnostics and "main" in args.studies and "noisy" in args.configs:
        from scripts.diagnose_noise_bias import run as noise_diagnostic

        log.info("noise-bias diagnostic (re-fits the noisy configuration on the saved paths) ...")
        t2 = time.perf_counter()
        noise_diagnostic(out, args.workers)
        diag_seconds = time.perf_counter() - t2

    meta = {
        "settings": base.to_dict(),
        "reps": args.reps,
        "studies": {s: [{"scenario": c.scenario, "configs": list(c.configs)} for c in study_cells(s, args)] for s in args.studies},
        "scenarios": args.scenarios,
        "scenario_parameters": {s: dataclasses.asdict(DEFAULT_CONFIGS[s]) for s in DEFAULT_CONFIGS},
        "configs": {c: {"description": get_config(c).description, "spec": repr(get_config(c).spec), "noise_dof": get_config(c).noise_dof}
                    for c in sorted({c for s in args.studies for cell in study_cells(s, args) for c in cell.configs})},
        "models": list(MODEL_ORDER),
        "wall_seconds": wall,
        "benchmark_seconds": bench_seconds,
        "diagnostic_seconds": diag_seconds,
        "workers": args.workers,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "platform": platform.platform(),
        "machine": platform.machine(),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "run_config.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("simulation finished in %.0fs (%d workers)", wall, args.workers)


def audit_invariants(df: pd.DataFrame) -> int:

    errors = df[df["status"] != "ok"]
    if len(errors):
        log.warning("%d (config, model, horizon) evaluations failed:", len(errors))
        for _, r in errors.drop_duplicates(["config", "scenario", "rep", "model"]).iterrows():
            log.warning("  %s/%s rep %d %s: %s", r["config"], r["scenario"], r["rep"], r["model"], str(r["error"]).splitlines()[-1])
    geo = df[(df["status"] == "ok") & df["model"].isin(GEOMETRIC)]
    bad = geo[(geo["valid_ratio"] < 1) | (geo["level_valid_ratio"] < 1)]
    if len(bad):
        log.error("INVARIANT VIOLATION: %d SPD-preserving-model evaluations produced non-SPD matrices", len(bad))
    else:
        log.info("SPD audit passed: all %d %s evaluations produced only SPD forecasts and levels", len(geo), "/".join(GEOMETRIC))
    return len(bad)


# ------------------------------ analysis. -----------------------------------
def fmt_ci(r: pd.Series) -> str:
    return f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"


def _tests_markdown(tests: pd.DataFrame) -> str:
    t = tests.copy()
    t["comparison"] = t["candidate"] + " vs " + t["reference"]
    t["cell"] = [
        f"{100 * a:+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}] (win {100 * w:.0f}%, p={p:.2g})"
        for a, lo, hi, w, p in zip(t["median_rel_improvement"], t["rel_ci_low"], t["rel_ci_high"], t["win_rate"], t["p_holm"])
    ]
    tt = t.pivot_table(index=["scenario", "horizon"], columns="comparison", values="cell", aggfunc="first").reset_index()
    return markdown_table(tt)


def _rmse_markdown(s: pd.DataFrame) -> str:
    table = s.assign(cell=s.apply(fmt_ci, axis=1)).pivot_table(index=["scenario", "horizon"], columns="model", values="cell", aggfunc="first")
    table = table[[m for m in MODEL_ORDER if m in table]].reset_index()
    valid = s[s["model"] == "euclidean"].set_index(["scenario", "horizon"])["valid_ratio"]
    table["Euclidean valid %"] = [100 * valid.get((sc, h), np.nan) for sc, h in zip(table["scenario"], table["horizon"])]
    return markdown_table(table, ".1f")


def write_summary(summary: pd.DataFrame, tests: pd.DataFrame, sens: pd.DataFrame, bench: pd.DataFrame, meta: dict, path: Path,
                  extra: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    configs = [c for c in CONFIGS if c in set(summary["config"])]
    lines = [
        "# Auto-generated results summary",
        "",
        f"Primary metric: `{PRIMARY_METRIC}` — AIRM RMSE over every test forecast; non-SPD Euclidean forecasts are "
        "repaired by eigenvalue clipping at 1e-3·λmax before scoring. Cells: mean over replications [95% bootstrap CI].",
        "",
        f"Models: {', '.join(MODEL_ORDER)}. Pre-registered contrasts: {', '.join(f'{a} vs {b}' for a, b in COMPARISONS)}; "
        "Holm family = every contrast × scenario × horizon of one configuration.",
        "",
        f"Replications per cell: {meta.get('reps')}; N = {meta['settings']['n_steps']}, "
        f"train = {meta['settings']['n_train']}, seed = {meta['settings']['base_seed']}.",
        "",
    ]
    for config in configs:
        s = summary[summary["config"] == config]
        lines += [f"## Configuration `{config}` — {CONFIGS[config].description}", "", _rmse_markdown(s), ""]
        lines += ["Paired tests (median relative improvement of candidate [95% bootstrap CI], win rate, Holm-adjusted Wilcoxon p):", "",
                  _tests_markdown(tests[tests["config"] == config]), ""]
    for scenario in ("rotation", "rotation_ou"):
        distortion = summary[summary["scenario"] == scenario][
            ["config", "horizon", "model", "valid_ratio", "trace_rel", "logdet_abs", "log_lmax_ratio", "logdet_err", "logtr_err", "frob_rmse", "le_rmse", "mean"]
        ].rename(columns={"mean": PRIMARY_METRIC})
        if len(distortion):
            lines += [f"## Spectral distortion — {scenario}", "",
                      "`trace_rel`: mean relative trace error; `logdet_abs`: mean |log det error| over SPD forecasts; "
                      "`log_lmax_ratio`: mean signed log-ratio of the largest eigenvalues (> 0 = dominant variance over-stated); "
                      "`logdet_err`, `logtr_err`: signed log-det (SPD forecasts) and log-trace errors.", "", markdown_table(distortion), ""]
    lines += ["## Sensitivity of Euclidean AIRM RMSE to the repair floor (LE and AIRM never repaired, for reference)", "", markdown_table(sens), ""]
    cost = bench.groupby(["scenario", "model"], sort=False)[["fit_seconds", "fit_evaluations", "filter_us_per_step", "forecast_us_per_matrix"]].median()
    lines += ["## Computational overhead (single-process benchmark, medians)", "", markdown_table(cost.reset_index()), ""]
    for study, (s, t) in extra.items():
        lines += [f"## Study `{study}`", ""]
        for config in s["config"].unique():
            lines += [f"### `{config}` — {get_config(config).description}", "", _rmse_markdown(s[s["config"] == config]), "",
                      _tests_markdown(t[t["config"] == config]), ""]
    path.write_text("\n".join(lines))


def scenario_features(out: Path) -> pd.DataFrame:
    records = []
    for scenario in MAIN_SCENARIOS:
        path = out / "data" / f"{scenario}.npz"
        if not path.exists():
            continue
        saved = np.load(path)
        S, lam = saved["Sigma"], saved["eigenvalues"]
        w = np.linalg.eigvalsh(S)
        cond = w[..., -1] / w[..., 0]
        steps = airm_distance(S[:, :-1], S[:, 1:])
        logdet = np.sum(np.log(w), axis=-1)
        trace = np.trace(S, axis1=-2, axis2=-1)
        rec = {
            "scenario": scenario, "n_paths": len(S), "n_steps": S.shape[1],
            "cond_median": np.median(cond), "cond_max": cond.max(),
            "eig_min": w.min(), "eig_max": w.max(),
            "step_median": np.median(steps), "step_p99": np.percentile(steps, 99),
            "trace_range": float(np.ptp(trace, axis=1).max()), "logdet_range": float(np.ptp(logdet, axis=1).max()),
            "share_lmin_below_0.05": float(np.mean(w[..., 0] < 0.05)), "share_lmin_below_0.01": float(np.mean(w[..., 0] < 0.01)),
            "max_step": steps.max(),
        }
        if scenario == "jump":
            ratio = lam / np.asarray(DEFAULT_CONFIGS["jump"].base)
            rec["share_any_above_3x_base"] = float(np.mean(np.any(ratio > 3.0, axis=-1)))
            rec["max_ratio_to_base"] = float(ratio.max())
        if scenario == "rotation_ou":
            base = np.asarray(DEFAULT_CONFIGS["rotation_ou"].eigenvalues)
            rec["log_eig_sd"] = float(np.std(np.log(lam / base)))
        records.append(rec)
    return pd.DataFrame.from_records(records)


def study_counts(out: Path, studies) -> pd.DataFrame:
    records = []
    for study in studies:
        df = pd.read_csv(out / "metrics" / metrics_file(study))
        ok = df[df["status"] == "ok"]
        geo = ok[ok["model"].isin(GEOMETRIC)]
        fits = df.drop_duplicates(["config", "scenario", "rep", "model"])
        records.append({
            "study": study, "n_fits": len(fits), "n_paths": df.drop_duplicates(["scenario", "rep"]).shape[0],
            "n_evaluations": len(df), "n_failed": int((df["status"] != "ok").sum()),
            "n_spd_preserving_evaluations": len(geo), "n_spd_preserving_forecasts": int(geo["n_forecasts"].sum()),
            "n_spd_violations": int(((geo["valid_ratio"] < 1) | (geo["level_valid_ratio"] < 1)).sum()),
        })
    out_df = pd.DataFrame.from_records(records)
    total = out_df.drop(columns="study").sum().to_dict()
    return pd.concat([out_df, pd.DataFrame([{"study": "all", **total}])], ignore_index=True)


def cost_summary(bench: pd.DataFrame) -> pd.DataFrame:
    med = bench.groupby(["scenario", "model"], sort=False)[["fit_seconds", "filter_us_per_step", "forecast_us_per_matrix"]].median().reset_index()
    e = med[med["model"] == "euclidean"].set_index("scenario")
    for col in ("fit_seconds", "filter_us_per_step", "forecast_us_per_matrix"):
        med[f"{col}_ratio"] = med[col].to_numpy() / e.loc[med["scenario"], col].to_numpy()
    med["dimension"] = [3 if s in MAIN_SCENARIOS else {"rotation_n10": 10, "dcc": 5}[s] for s in med["scenario"]]
    return med


def analyse(out: Path) -> int:
    meta = json.loads((out / "run_config.json").read_text())
    if not meta.get("studies"):  # real data only: nothing simulated to analyse
        return 0
    summary_dir = out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    violations = 0
    extra = {}
    for study in meta["studies"]:
        df = pd.read_csv(out / "metrics" / metrics_file(study))
        log.info("study %s:", study)
        violations += audit_invariants(df)
        summary, tests = summary_table(df), pairwise_tests(df)
        prefix = "" if study == "main" else f"{study}_"
        summary.to_csv(summary_dir / f"{prefix}summary_table.csv", index=False)
        tests.to_csv(summary_dir / f"{prefix}pairwise_tests.csv", index=False)
        volume_bias(df).to_csv(summary_dir / f"{prefix}volume_bias.csv", index=False)
        params_summary(df).to_csv(summary_dir / f"{prefix}params_summary.csv", index=False)
        if study == "main":
            main_df, main_summary, main_tests = df, summary, tests
        else:
            extra[study] = (summary, tests)
    bench = pd.read_csv(out / "metrics" / "compute_benchmark.csv")
    cost_summary(bench).to_csv(summary_dir / "cost_summary.csv", index=False)
    study_counts(out, meta["studies"]).to_csv(summary_dir / "study_counts.csv", index=False)
    if (out / "realdata" / "per_origin.csv.gz").exists():
        # Re-summarise the saved per-origin losses (deterministic: same bootstrap seed), so that the real-data
        # tables can be rebuilt with --plots-only without re-fitting anything.
        from src.metrics.summary import COMPARISONS as _C
        from src.realdata_experiment import summarise as _summarise
        real_summary, real_tests = _summarise(pd.read_csv(out / "realdata" / "per_origin.csv.gz"), _C)
        real_summary.to_csv(summary_dir / "realdata_summary.csv", index=False)
        real_tests.to_csv(summary_dir / "realdata_tests.csv", index=False)
        write_realdata_features(summary_dir)
    if "main" not in meta["studies"]:
        return violations
    sens = floor_sensitivity(main_df, tuple(meta["settings"]["floors"]))
    sens.to_csv(summary_dir / "repair_floor_sensitivity.csv", index=False)
    features = scenario_features(out)
    if len(features):
        features.to_csv(summary_dir / "scenario_features.csv", index=False)
    write_summary(main_summary, main_tests, sens, bench, meta, summary_dir / "summary.md", extra)
    log.info("tables written to %s", summary_dir)
    return violations


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(args.out)
    log.info("log file: %s", log_path)
    t0 = time.perf_counter()
    real = None
    if args.benchmark_only:
        settings = RunSettings(n_steps=args.n_steps, n_train=args.n_train, base_seed=args.seed)
        log.info("re-measuring the single-process cost benchmark ...")
        bench = pd.DataFrame(benchmark_models(BENCH_SCENARIOS, BENCH_REPS, settings))
        bench.to_csv(args.out / "metrics" / "compute_benchmark.csv", index=False)
        meta_path = args.out / "run_config.json"
        meta = json.loads(meta_path.read_text())
        meta["benchmark_remeasured"] = datetime.now().isoformat(timespec="seconds")
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        args.plots_only = True
    if not args.plots_only:
        settings = RunSettings(n_steps=args.n_steps, n_train=args.n_train, base_seed=args.seed)
        mc = [s for s in args.studies if s != REAL]
        if mc:
            run_monte_carlo(argparse.Namespace(**{**vars(args), "studies": mc}), settings)
        if REAL in args.studies:
            real = run_realdata(args.out, args.workers)
            meta_path = args.out / "run_config.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"studies": {}}
            meta["realdata"] = {"seconds": real["seconds"], "window": 360, "refit_every": 12, "horizons": [1, 3, 6, 12]}
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
    violations = analyse(args.out) + (real["violations"] if real else 0)
    log.info("total elapsed %.0fs", time.perf_counter() - t0)
    if violations:
        sys.exit(2)  


if __name__ == "__main__":
    main()
