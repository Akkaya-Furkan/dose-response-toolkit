import json
import numpy as np
import pytest

from doseresponse import analyze, fit_4pl, fourpl
from doseresponse.synth import log_dilution, simulate
from doseresponse.report import html_report, pdf_report
from doseresponse.cli import main

CONC = log_dilution(100, 10)


def codes(A):
    return {c for c, _, _ in A.flags}


def test_noiseless_recovery_exact():
    c, y = simulate(CONC, ic50=2.0, hill=1.5, noise_sd=0.0)
    f = fit_4pl(np.log10(c), y)
    assert f.params == pytest.approx([100, 0, np.log10(2.0), 1.5], abs=1e-4)


@pytest.mark.parametrize("ic50,hill", [(0.1, 1.0), (3, 0.8), (3, 2.0), (20, 1.0)])
def test_noisy_recovery(ic50, hill):
    c, y = simulate(CONC, ic50=ic50, hill=hill, noise_sd=3.0, seed=5)
    A = analyze(c, y, n_boot=100)
    assert A.ic50 is not None
    assert abs(A.logic50 - np.log10(ic50)) < 0.25


def test_increasing_curve():
    c, y = simulate(CONC, y0=0, yinf=100, ic50=3, noise_sd=2, seed=2)
    A = analyze(c, y, n_boot=100)
    assert A.ic50 == pytest.approx(3, rel=0.3)


def test_unbracketed_ic50_is_not_reported():
    # noiseless: true IC50 (1000 uM) far above the 0.01-100 uM range, shallow slope
    x = np.linspace(-2, 2, 10)
    A = analyze(10 ** x, fourpl(x, 100, 0, 3, 0.2), n_boot=100)
    assert A.ic50 is None and A.status == "undefined"
    assert codes(A) & {"PLATEAU_UNIDENTIFIED", "IC50_NOT_BRACKETED", "IC50_CI_NOT_BRACKETED"}
    assert A.to_dict()["params"]["logic50"] is None


def test_unbracketed_noisy_not_reported():
    c, y = simulate(CONC, ic50=1e4, noise_sd=1.0, seed=3)
    A = analyze(c, y, n_boot=100)
    assert A.ic50 is None and A.status == "undefined"


def test_undefined_json_does_not_leak_ic50():
    x = np.linspace(-2, 2, 10)
    A = analyze(10 ** x, fourpl(x, 100, 0, 3, 1), n_boot=100)
    assert A.ic50 is None
    assert A.to_dict()["params"]["logic50"] is None


def test_scale_invariance():
    c, y = simulate(CONC, ic50=3, noise_sd=2.0, seed=8)
    a = fit_4pl(np.log10(c), y)
    b = fit_4pl(np.log10(c), y * 1e-6)
    assert a.logic50 == pytest.approx(b.logic50, abs=1e-3)
    assert a.hill == pytest.approx(b.hill, rel=1e-2)


def test_constant_response():
    for v in (0.0, 1.0, 100.0):
        A = analyze(CONC, np.full(10, v))
        assert A.ic50 is None and "NO_ACTIVITY" in codes(A)


def test_n_boot_validated():
    from doseresponse import bootstrap_ci
    c, y = simulate(CONC, ic50=3, noise_sd=2.0)
    with pytest.raises(ValueError):
        bootstrap_ci(np.log10(c), y, n_boot=5)


def test_missing_column_raises(tmp_path):
    from doseresponse.io import read_long_csv
    f = tmp_path / "a.csv"
    f.write_text("id,c,r\na,1,2\na,2\n")
    with pytest.raises(KeyError):
        read_long_csv(f, "id", "nope", "r")
    assert len(read_long_csv(f, "id", "c", "r")["a"][0]) == 1  # truncated record skipped


def test_ic50_below_range_not_reported():
    c, y = simulate(CONC, ic50=1e-4, noise_sd=1.0, seed=3)
    A = analyze(c, y, n_boot=100)
    assert A.ic50 is None and A.status == "undefined"


def test_flat_data_flagged_no_activity():
    rng = np.random.default_rng(0)
    A = analyze(CONC, 50 + rng.normal(0, 3, len(CONC)), n_boot=100)
    assert A.ic50 is None and "NO_ACTIVITY" in codes(A)


def test_too_few_points():
    A = analyze(CONC[:3], [100, 50, 0])
    assert A.ic50 is None and "TOO_FEW_POINTS" in codes(A)


def test_missing_plateau_flag():
    # IC50 inside range, upper end of the range stops short of the high-dose plateau
    x = np.linspace(-2, 1.3, 9)
    y = fourpl(x, 100, 0, 0.8, 1.0)
    A = analyze(10 ** x, y + np.random.default_rng(1).normal(0, 0.5, 9), n_boot=100)
    assert A.ic50 is None or "MISSING_PLATEAU_HIGH_DOSE" in codes(A)
    f = fourpl(np.array([x[-1]]), 100, 0, 0.8, 1.0)[0]
    assert f > 10  # the premise: curve has not reached the plateau


def test_poor_fit_never_ok():
    c, y = simulate(CONC, ic50=3, noise_sd=30.0, seed=9)
    A = analyze(c, y, n_boot=100)
    assert A.status != "ok"


def test_nonpositive_conc_dropped():
    c, y = simulate(CONC, ic50=3, noise_sd=2.0)
    A = analyze(np.r_[0.0, c], np.r_[100.0, y], n_boot=100)
    assert A.dropped_nonpositive == 1 and "NONPOSITIVE_CONC" in codes(A)


def test_bootstrap_ci_contains_truth_typically():
    hits = 0
    for s in range(20):
        c, y = simulate(CONC, ic50=3, noise_sd=5, seed=100 + s)
        A = analyze(c, y, n_boot=100, seed=s)
        assert A.logic50_ci is not None
        lo, hi = A.logic50_ci
        hits += lo <= np.log10(3) <= hi
    assert hits >= 16  # nominal 90-95%; loose lower bound for 20 datasets (see experiments/recovery.py for the real study)


def test_reports_and_cli(tmp_path):
    c, y = simulate(CONC, ic50=3, noise_sd=4)
    items = [("good", analyze(c, y, n_boot=100)), ("flat", analyze(CONC, np.full(10, 50.0)))]
    h = html_report(items)
    assert "undefined (not reported)" in h and "<img" in h
    pdf_report(items, tmp_path / "r.pdf")
    assert (tmp_path / "r.pdf").stat().st_size > 1000
    csv = tmp_path / "d.csv"
    csv.write_text("id,c,r\n" + "\n".join(f"a,{ci},{yi}" for ci, yi in zip(c, y)))
    main([str(csv), "--id", "id", "--conc", "c", "--resp", "r", "--boot", "100",
          "--json", str(tmp_path / "o.json"), "--html", str(tmp_path / "o.html")])
    assert json.load(open(tmp_path / "o.json"))["a"]["ic50"] is not None
