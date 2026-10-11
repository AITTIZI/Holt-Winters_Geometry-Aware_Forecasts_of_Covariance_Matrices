# Real data: provenance and transformations

Used by the real-data experiment code in `src/data/realdata.py` and `src/realdata_experiment.py`; 
run by `python run_experiments.py`, study `realdata`.

## Source
| Dataset key | File | URL | n |
|---|---|---|---|
| `ff5` (primary) | `5_Industry_Portfolios_daily_CSV.zip` | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/5_Industry_Portfolios_daily_CSV.zip | 5 (Cnsmr, Manuf, HiTec, Hlth, Other) |
| `ff10` (secondary) | `10_Industry_Portfolios_daily_CSV.zip` | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Industry_Portfolios_daily_CSV.zip | 10 (NoDur, Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils, Other) |

* Provider: Kenneth R. French Data Library, Tuck School of Business,
  Dartmouth (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
* Vintage: the files state "This file was created using the 202608 CRSP
  database". Daily data run from 1926-07-01 to 2026-08-31.
* Access: downloaded by `src/data/realdata.py` on the date and with the SHA-256
  checksums recorded in `results/summary/realdata_provenance.csv`. The original
  download for the manuscript is listed below.
* Terms: the library page states only "Copyright Eugene F. Fama and Kenneth R.
  French". It gives no explicit licence. **TODO (authors): confirm that the raw
  files may be redistributed with the code before putting `data/raw/` in a
  public repository.** Otherwise keep `data/raw/` out of the repository, and let
  the script download the files (the checksums identify the exact vintage).

## Transformations (in this order)
1. Read the block "Average Value Weighted Returns -- Daily" (percent). The
   equal-weighted block is not used. Missing values (−99.99 or −999) would abort
   the run. There are none in this vintage.
2. Daily log returns `r_d = 100 · log(1 + R_d / 100)`.
3. Monthly realised covariance `C_m = Σ_{d ∈ m} r_d r_dᵀ` (not demeaned; units
   %² per month), one matrix per calendar month: 1,202 months, 1926-07 to
   2026-08, with 15 to 27 trading days per month. Every matrix is checked to be
   SPD.
4. No further filtering, winsorising or rescaling.

## Original download for the manuscript
See `results/summary/realdata_provenance.csv`
