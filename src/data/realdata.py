from __future__ import annotations

import hashlib
import http.client
import shutil
import subprocess
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geometry.spd import SPDViolationError, is_spd

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
DATASETS = {
    "ff5": "5_Industry_Portfolios_daily_CSV.zip",
    "ff10": "10_Industry_Portfolios_daily_CSV.zip",
}

BLOCK = "Average Value Weighted Returns -- Daily"
MISSING = (-99.99, -999.0)


@dataclass
class MonthlyCovariances:
    name: str
    months: np.ndarray  # (M,) numpy datetime64[M]
    industries: tuple[str, ...]
    C: np.ndarray  # (M, n, n) realised covariance, %^2 per month
    n_days: np.ndarray  # (M,) trading days per month
    source_note: str  # first line of the source file (CRSP vintage)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(name: str, raw_dir: Path, attempts: int = 2) -> Path:
    """Download a French Data Library dataset to ``raw_dir`` and return the path to the zip file."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / DATASETS[name]
    if path.exists():
        return path
    tmp = path.with_suffix(".part")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(FRENCH_URL + DATASETS[name], timeout=60) as response:
                tmp.write_bytes(response.read())
            if not zipfile.is_zipfile(tmp):
                raise OSError(f"{tmp.name} is not a valid zip archive")
            tmp.replace(path)
            return path
        except (OSError, http.client.HTTPException) as exc:  
            last_error = exc
            tmp.unlink(missing_ok=True)
            time.sleep(2.0 * (attempt + 1))

    if shutil.which("curl"):
        result = subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", str(tmp), FRENCH_URL + DATASETS[name]], capture_output=True, text=True)
        if result.returncode == 0 and zipfile.is_zipfile(tmp):
            tmp.replace(path)
            return path
        tmp.unlink(missing_ok=True)
        last_error = RuntimeError(f"curl: {result.stderr.strip()}")
    raise RuntimeError(f"could not download {DATASETS[name]} after {attempts} attempts: {last_error!r}")


def read_value_weighted_daily(source: Path | str | bytes) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, str]:
    if isinstance(source, bytes):
        text = source.decode("latin-1")
    else:
        source = Path(source)
        if source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source) as zf:
                member = next(m for m in zf.namelist() if m.lower().endswith(".csv"))
                text = zf.read(member).decode("latin-1")
        else:
            text = source.read_text(encoding="latin-1")
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == BLOCK)
    except StopIteration:
        raise ValueError(f"block {BLOCK!r} not found") from None
    header = [h.strip() for h in lines[start + 1].split(",")[1:]]
    dates, rows = [], []
    for line in lines[start + 2 :]:
        if not line.strip():
            break
        fields = [f.strip() for f in line.split(",")]
        dates.append(np.datetime64(f"{fields[0][:4]}-{fields[0][4:6]}-{fields[0][6:8]}"))
        rows.append([float(x) for x in fields[1:]])
    returns = np.asarray(rows, dtype=float)
    if returns.shape[1] != len(header):
        raise ValueError("row width does not match the header")
    if np.any(np.isin(returns, MISSING)) or not np.all(np.isfinite(returns)):
        raise ValueError("the value-weighted daily block contains missing values")
    if np.any(np.diff(np.asarray(dates)) <= np.timedelta64(0, "D")):
        raise ValueError("dates are not strictly increasing")
    return np.asarray(dates), tuple(header), returns, lines[0].strip()


def monthly_realized_covariance(dates: np.ndarray, returns_pct: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = 100.0 * np.log1p(np.asarray(returns_pct, dtype=float) / 100.0)
    month = np.asarray(dates).astype("datetime64[M]")
    months, idx, counts = np.unique(month, return_inverse=True, return_counts=True)
    n = r.shape[1]
    C = np.zeros((len(months), n, n))
    np.add.at(C, idx, r[:, :, None] * r[:, None, :])
    C = 0.5 * (C + np.swapaxes(C, -1, -2))
    return months, C, counts


def load_monthly(name: str, raw_dir: Path) -> MonthlyCovariances:
    path = download(name, raw_dir)
    dates, industries, returns, note = read_value_weighted_daily(path)
    months, C, n_days = monthly_realized_covariance(dates, returns)
    ok = is_spd(C)
    if not ok.all():
        raise SPDViolationError(f"{name}: {np.count_nonzero(~ok)} monthly realised covariances are not SPD")
    return MonthlyCovariances(name, months, industries, C, n_days, note)
