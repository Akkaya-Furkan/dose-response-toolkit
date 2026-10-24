# Dose–response toolkit — how reliable is an IC50?

[![ci](https://github.com/Akkaya-Furkan/dose-response-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Akkaya-Furkan/dose-response-toolkit/actions/workflows/ci.yml)

A small Python package that fits four-parameter logistic (4PL) curves, puts a bootstrap
confidence interval on the IC50, and **refuses to report an IC50 that the data cannot support**.

## Question

How reliable are IC50 estimates from 4PL fits — in particular, how often is a reported
interval right, and when should the number not be reported at all?

## Approach

* **Model** (`model.py`): `y = yinf + (y0 − yinf) / (1 + 10^(hill·(x − logIC50)))`, x = log10 concentration.
  Bounded least squares (SciPy) with multi-start initialisation; response rescaled internally so results do not depend on units.
* **Bootstrap** (`bootstrap.py`): percentile 95% CI by residual bootstrap (default) or case resampling; warm-started refits plus two extra starts.
* **QC** (`qc.py`). An IC50 is *withheld* (status `undefined`, no number in the report or JSON) if: fewer than 5 distinct positive concentrations; constant response or no significant concentration-response (F-test against a flat line); fitted IC50 outside the tested range; fitted plateaus more than half the observed response span outside the data; the 95% bootstrap interval leaves the tested range; or fewer than 80% of bootstrap fits are usable. Other problems give flags with status `warn`: `MISSING_PLATEAU_*`, `EXTREME_HILL` (outside 0.5–2.5), `POOR_FIT` (RMSE > 15% of span), `WIDE_CI` (> 1 log10), `PARAM_AT_BOUND`, `FEW_CONCENTRATIONS`, `OPTIMIZER_NOT_CONVERGED`.
* **Reports**: self-contained HTML and multi-page PDF; JSON from the CLI.
* **Evidence** (`experiments/`): parameter recovery on synthetic data with known truth, a null (pure noise) check, and a run on public PubChem data.

```bash
pip install -e '.[dev]'
drtk data/pubchem_aid411_subset.csv --id sid --conc conc_uM --resp response_pct \
     --html report.html --pdf report.pdf --json out.json --limit 12
pytest
```

```python
from doseresponse import analyze
A = analyze(conc, response, unit="uM")      # A.status, A.ic50, A.ic50_ci, A.flags
```

## Data

* **Synthetic**: 10-point 3-fold dilution, 0.005–100 µM, y0=100, yinf=0, Gaussian noise; true IC50, noise and Hill slope varied (`experiments/recovery.py`).
* **Public**: [PubChem BioAssay AID 411](https://pubchem.ncbi.nlm.nih.gov/bioassay/411) (NCGC qHTS, firefly luciferase inhibition). 300 substances sampled with a fixed seed (100 Active, 100 Inconclusive, 100 Inactive per the depositor), 7 concentrations each, in `data/pubchem_aid411_subset.csv`; built by `scripts/make_pubchem_subset.py` from the PubChem FTP file.

## Results

All numbers below are produced by the scripts in `experiments/` and stored in `results/`.

**Synthetic recovery** (100 datasets × 100 bootstrap resamples per scenario; `results/recovery.json`; coverage standard error ≈ 3 points). Residual bootstrap, default noise SD 5 on a 100-unit span, Hill 1:

| true IC50 (µM) | in range | IC50 reported | median bias (log10) | conditional CI coverage |
|---|---|---|---|---|
| 0.001 | no | 0% | – | – |
| 0.01 | edge | 15% | +0.33 | 60% |
| 0.03 | yes | 50% | +0.08 | 88% |
| 0.3 | yes | 100% | −0.01 | 90% |
| 3 | yes | 98% | −0.01 | 91% |
| 30 | edge | 25% | −0.24 | 80% |
| 300 | no | 0% | – | – |
| 10 000 | no | 0% | – | – |

* With the IC50 well inside the range, the median error is ≈0 and RMSE ≈ 0.09 log10 (≈ 23%); the nominal 95% interval covered the truth in about 90–91% of the datasets that were reported — **slightly anti-conservative**.
* Higher noise hurts quickly: at noise SD 10 and 20, 69% and 33% of datasets still yield a number, and coverage among those falls to 86% and 82%. Shallow curves (Hill 0.5) are the worst case: 37% reported, coverage 73%.
* Truth outside the tested range (0.001, 300, 10 000 µM): residual bootstrap produced a number in 0 of 300 datasets; case bootstrap in 2 of 300 (both with status `warn`, e.g. one 300 µM dataset reported with −0.88 log10 error). Suppression is strong but not absolute.
* Edge cases (IC50 within ~0.5 log of the range limit) are the weak spot: where a number is reported, coverage is 60–88%.
* Case bootstrap with 10 distinct doses often gives degenerate resamples; the tool then withholds the IC50 (61% reported for the 3 µM scenario) — it is the conservative option, residual bootstrap is the default.
* **Null check** (`results/null_calibration.json`): on 300 flat-noise datasets, 1.7% passed the activity gate and 0.3% (1 dataset) ended with a reported IC50.

**PubChem AID 411** (300 substances, `results/pubchem_summary.json`):

| depositor outcome | ok | warn | undefined (no IC50) |
|---|---|---|---|
| Active (100) | 31 | 32 | 37 |
| Inconclusive (100) | 11 | 22 | 67 |
| Inactive (100) | 0 | 2 | 98 |

* Where this tool reports an IC50 and the depositor also lists a fitted AC50 (96 substances), the two agree closely: median |difference| 0.037 log10, all within 0.3 log10, Pearson r = 0.98. This is agreement between two Hill fits to the same points, not validation against a ground truth.
* The depositor lists an AC50 for 104 substances for which this tool returns *undefined* (mostly extrapolated or partial curves; these are exactly the cases the tool is designed to refuse). Median reported 95% CI width: 0.19 log10.
* Example report with 11 curves (first of each category, not random): [docs/example_report.html](docs/example_report.html) and [PDF](docs/example_report.pdf).

## Limitations

* **Not done:** no hierarchical / replicate-level error model; no outlier removal; no Bayesian intervals; no multiple-testing treatment across a plate; no 3-parameter (fixed-slope) or biphasic models; no global fits across plates.
* Residual bootstrap assumes homoscedastic noise; heavy-tailed or proportional noise is not modelled. Synthetic noise is Gaussian and homoscedastic, so real-data coverage may be worse than reported here.
* Reported coverage is **conditional on the IC50 being reported**; it excludes suppressed datasets and so is not the unconditional coverage of the full procedure. It is also measured on a single design (10 points, 3-fold) with 100 datasets per scenario.
* The F-test activity gate is approximate (the midpoint and slope are unidentified under the null, so the F(3, n−4) reference is not exact); its practical false-pass rate was checked by simulation only (1.7%, one noise level, one design).
* The thresholds (Hill 0.5–2.5, RMSE 15% of span, CI width 1 log10, plateau 10%, 0.5-span plateau rule) are pragmatic defaults, not tuned or validated.
* If the dose range does not bracket the IC50 the estimate is undefined: the tool flags it and prints no number. This guards against fits whose midpoint is extrapolated; it can still err near the range limits (see edge cases above).
* The AID 411 assay is a luciferase counter-screen; results show how the tool behaves on public qHTS data, not any biological conclusion. Only 300 of ~72 000 substances were analysed.
* IC50 here is the 4PL midpoint (relative IC50), not the concentration giving 50% of control response.

## Layout

`src/doseresponse/` package · `tests/` (21 tests) · `experiments/` and `results/` · `data/` · `docs/` example report · `.github/workflows/ci.yml`.
