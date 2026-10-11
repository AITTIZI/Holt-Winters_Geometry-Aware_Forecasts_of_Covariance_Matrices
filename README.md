# Does Respecting Matrix Geometry Improve Holt–Winters Forecasts of Covariance Matrices?

## What the project does
Holt–Winters smoothing of covariance matrices is usually done element-wise,
although extrapolating a trend can leave the cone of positive-definite
matrices. The project asks whether respecting the geometry of the SPD manifold
improves Holt–Winters forecasting, and, if so, whether the curved
affine-invariant geometry is needed or the flat Log-Euclidean geometry is
enough. It compares three versions of one Holt–Winters recursion:

* **Euclidean HW**: element-wise smoothing under the flat ambient metric
  (baseline, not SPD-preserving);
* **Log-Euclidean (LE) HW**: the flat metric of the matrix logarithm
  (SPD-preserving);
* **AIRM HW**: the curved affine-invariant metric, with the trend carried by
  parallel transport (SPD-preserving).

They are compared in paired Monte Carlo studies (Scenarios A, A′, B and C; a
common-fitting-criterion study; a sensitivity sweep; an n = 10 run; a
DCC-GARCH process) and in a rolling-origin application to monthly realised
covariances of the Kenneth R. French industry portfolios.

## Status
* The study is complete. The final run is in `results/`, and an independent
  re-run reproduced every non-timing value of `results/summary/*`.


## Setup and running
The reported run used Python 3.13.5 and the exact package versions pinned in
`requirements.txt`; only these are known to reproduce `results/summary/*`.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_experiments.py        
```


## Repository layout
| Path | Content |
|---|---|
| `src/geometry/` | SPD primitives (eigen-based `expm`/`logm`/powers, SPD tests, target-blind repair), AIRM (Exp/Log, geodesics, parallel transport, distance), Log-Euclidean distance and mean, SO(3) |
| `src/optimization/` | Karcher mean (Riemannian gradient descent), smoothing-parameter estimation |
| `src/models/` | `EuclideanHoltWinters`, `LogEuclideanHoltWinters`, `AIRMHoltWinters` (trend, damping, seasonality; own-geometry or common fitting criterion) |
| `src/data/` | SO(3)/SO(n) random walks, exact CIR, exact log-OU, Merton jump-diffusion, DCC-GARCH, Wishart observation noise, the scenarios and their variants; `realdata.py`: French industry portfolios to monthly realised covariances |
| `src/metrics/` | validity, AIRM/LE/Frobenius errors, spectral and signed volume diagnostics, repair-floor sensitivity, paired Wilcoxon + Holm (three contrasts), bootstrap CIs |
| `src/realdata_experiment.py` | rolling-origin evaluation, Stein loss, Diebold–Mariano (HLN, Newey–West), moving-block bootstrap |
| `src/visualization/` | plotting functions for the manuscript figures |
| `src/experiment.py` | one Monte Carlo replication; the registry of studies; single-process cost benchmark |
| `run_experiments.py` | orchestration: parallel simulation, real data, audits, summary tables |
| `scripts/` | noise-bias diagnostic (`diagnose_noise_bias.py`); manuscript number check |
| `data/` | real-data provenance (`README.md`); downloaded files in `data/raw/` |

