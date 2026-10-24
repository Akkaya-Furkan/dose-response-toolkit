"""Parameter recovery and CI coverage on synthetic data.

Design: 10-point 3-fold dilution series, top concentration 100 uM (tested range 0.005-100 uM,
log10 -2.3..2), y0=100, yinf=0, hill=1, Gaussian noise. We vary the true IC50 (including values
outside the tested range) and the noise SD, and compare case vs residual bootstrap.
Run: python experiments/recovery.py [n_datasets] [n_boot]
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from doseresponse import analyze
from doseresponse.synth import log_dilution, simulate

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 200
CONC = log_dilution(100, 10)
XMIN, XMAX = np.log10(CONC[0]), np.log10(CONC[-1])


def one(args):
    ic50, sd, hill, method, i = args
    c, y = simulate(CONC, ic50=ic50, hill=hill, noise_sd=sd, seed=1000 * i + 7)
    A = analyze(c, y, n_boot=NB, boot_method=method, seed=i)
    return {"status": A.status, "logic50": A.logic50, "ci": A.logic50_ci, "hill": float(A.fit.hill) if A.fit else None,
            "codes": [f[0] for f in A.flags]}


def summarise(res, true_log):
    rep = [r for r in res if r["logic50"] is not None]
    out = {"n": len(res), "reported": len(rep), "reported_frac": len(rep) / len(res)}
    if rep:
        est = np.array([r["logic50"] for r in rep])
        out["bias_log10"] = float(np.median(est - true_log))
        out["rmse_log10"] = float(np.sqrt(np.mean((est - true_log) ** 2)))
        withci = [r for r in rep if r["ci"]]
        if withci:
            cov = [r["ci"][0] <= true_log <= r["ci"][1] for r in withci]
            out["conditional_coverage95"] = float(np.mean(cov))
            out["n_with_ci"] = len(withci)
            out["covered_and_reported_frac_of_all"] = float(np.sum(cov)) / len(res)
            out["median_ci_width_log10"] = float(np.median([r["ci"][1] - r["ci"][0] for r in withci]))
        flagged = [r for r in rep if r["status"] == "warn"]
        out["warn_frac_of_reported"] = len(flagged) / len(rep)
        # conditional coverage among unflagged (status ok) fits only
        okr = [r for r in withci if r["status"] == "ok"]
        out["n_ok"] = len(okr)
        if okr:
            out["conditional_coverage95_ok_only"] = float(np.mean([r["ci"][0] <= true_log <= r["ci"][1] for r in okr]))
    out["true_in_range"] = bool(XMIN <= true_log <= XMAX)
    return out


def main():
    scen = []
    for method in ("case", "residual"):
        for ic50 in (0.001, 0.01, 0.03, 0.3, 3, 30, 300, 1e4):
            scen.append((ic50, 5.0, 1.0, method))
        for sd in (2.0, 10.0, 20.0):
            scen.append((3, sd, 1.0, method))
        for hill in (0.5, 2.0, 4.0):
            scen.append((3, 5.0, hill, method))
    results = []
    with ProcessPoolExecutor() as ex:
        for ic50, sd, hill, method in scen:
            res = list(ex.map(one, [(ic50, sd, hill, method, i) for i in range(N)], chunksize=10))
            s = summarise(res, np.log10(ic50))
            s.update(true_ic50=ic50, noise_sd=sd, hill=hill, method=method)
            results.append(s)
            print(json.dumps(s), flush=True)
    json.dump({"n_datasets": N, "n_boot": NB, "range_uM": [float(CONC[0]), float(CONC[-1])], "scenarios": results},
              open("results/recovery.json", "w"), indent=1)


if __name__ == "__main__":
    main()
