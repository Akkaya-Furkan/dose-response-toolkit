"""How often does the activity gate (F-test vs flat line) let pure noise through?

Flat responses (mean 50, Gaussian noise) on the 10-point design; counts datasets for which
the analysis does NOT return NO_ACTIVITY, and datasets for which an IC50 is finally reported.
"""
import json
import numpy as np
from doseresponse import analyze
from doseresponse.synth import log_dilution

conc = log_dilution(100, 10)
rng = np.random.default_rng(0)
N = 300
gate_pass = reported = 0
for i in range(N):
    A = analyze(conc, 50 + rng.normal(0, 5, len(conc)), n_boot=100, seed=i)
    if not any(c == "NO_ACTIVITY" for c, _, _ in A.flags):
        gate_pass += 1
    reported += A.ic50 is not None
out = {"n": N, "gate_pass_frac": gate_pass / N, "ic50_reported_frac": reported / N}
json.dump(out, open("results/null_calibration.json", "w"), indent=1)
print(out)
