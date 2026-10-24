"""End-to-end analysis with QC flags.

Policy: the tool never reports an IC50 when the data cannot support one. IC50
is reported as ``None`` (status ``undefined``) if
  * there are too few distinct concentrations,
  * the fit is not significantly better than a flat line (no activity),
  * the fitted IC50 lies outside the tested concentration range (not bracketed),
  * the 95% bootstrap interval of the IC50 extends beyond the tested range.
Other problems produce flags with status ``warn``; the number is still shown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy import stats

from .model import fit_4pl, fourpl
from .bootstrap import bootstrap_ci

MIN_CONC = 5
HILL_LOW, HILL_HIGH = 0.5, 2.5
PLATEAU_FRAC = 0.10      # curve must reach within 10% of the asymptote at the dose extremes
RMSE_SPAN_MAX = 0.15
CI_WIDTH_LOG_MAX = 1.0   # 95% CI wider than 1 log10 unit is flagged


@dataclass
class Analysis:
    x: np.ndarray
    y: np.ndarray
    fit: object = None
    boot: dict | None = None
    status: str = "undefined"          # ok | warn | undefined
    flags: list = field(default_factory=list)   # (code, severity, message)
    ic50: float | None = None          # same unit as the input concentration
    ic50_ci: tuple | None = None
    logic50: float | None = None
    logic50_ci: tuple | None = None
    r2: float | None = None
    rmse_over_span: float | None = None
    unit: str = ""
    dropped_nonpositive: int = 0

    def add(self, code, severity, msg):
        self.flags.append((code, severity, msg))

    def to_dict(self):
        f = self.fit
        return {
            "status": self.status, "ic50": self.ic50, "ic50_ci": self.ic50_ci,
            "unit": self.unit,
            "params": None if f is None else {**dict(zip(("y0", "yinf", "logic50", "hill"), map(float, f.params))),
                                              **({"logic50": None} if self.ic50 is None else {})},
            "r2": self.r2, "rmse_over_span": self.rmse_over_span,
            "flags": [{"code": c, "severity": s, "message": m} for c, s, m in self.flags],
            "n_points": int(len(self.x)), "n_conc": int(len(np.unique(self.x))),
        }


def analyze(conc, resp, unit="uM", n_boot=500, boot_method="residual", seed=0, do_boot=True) -> Analysis:
    conc = np.asarray(conc, float)
    resp = np.asarray(resp, float)
    ok = np.isfinite(conc) & np.isfinite(resp)
    conc, resp = conc[ok], resp[ok]
    pos = conc > 0
    a_dropped = int((~pos).sum())
    conc, resp = conc[pos], resp[pos]
    x = np.log10(conc)
    A = Analysis(x=x, y=resp, unit=unit, dropped_nonpositive=a_dropped)
    if a_dropped:
        A.add("NONPOSITIVE_CONC", "info", f"{a_dropped} point(s) with concentration <= 0 excluded (not representable on a log axis).")
    nconc = len(np.unique(x))
    if np.ptp(resp) <= 0:
        A.add("NO_ACTIVITY", "fatal", "Response is constant; no concentration-response detected.")
        return A
    if nconc < MIN_CONC or len(x) < 6:
        A.add("TOO_FEW_POINTS", "fatal", f"Only {nconc} distinct positive concentrations ({len(x)} points); need >= {MIN_CONC} to fit 4 parameters.")
        return A
    f = fit_4pl(x, resp)
    A.fit = f
    if not np.isfinite(f.sse):
        A.add("FIT_FAILED", "fatal", "Optimizer did not converge.")
        return A
    sst = float(((resp - resp.mean()) ** 2).sum())
    A.r2 = 1 - f.sse / sst if sst > 0 else 0.0
    span = abs(f.y0 - f.yinf)
    A.rmse_over_span = f.rmse / span if span > 0 else float("inf")

    # activity: extra-sum-of-squares F test against a constant model
    dfn, dfd = 3, len(x) - 4
    if dfd > 0 and f.sse > 0 and sst > f.sse:
        F = ((sst - f.sse) / dfn) / (f.sse / dfd)
        p = float(stats.f.sf(F, dfn, dfd))
    else:
        p = 1.0 if sst <= f.sse else 0.0
    if p > 0.05:
        A.add("NO_ACTIVITY", "fatal", f"4PL not significantly better than a flat line (F-test p={p:.3g}); no concentration-response detected.")
        return A

    ylo, yhi = resp.min(), resp.max()
    ext = 0.5 * (yhi - ylo)
    for nm, v in (("zero-dose", f.y0), ("high-dose", f.yinf)):
        if v < ylo - ext or v > yhi + ext:
            A.add("PLATEAU_UNIDENTIFIED", "fatal",
                  f"Fitted {nm} plateau ({v:.3g}) lies more than half the observed response span outside the data ({ylo:.3g}-{yhi:.3g}); the curve midpoint, and hence the IC50, is not identified.")
            return A
    if not f.success:
        A.add("OPTIMIZER_NOT_CONVERGED", "warn", "Optimizer stopped before converging; treat the fit with caution.")

    # bracketing
    xmin, xmax = x.min(), x.max()
    if not (xmin <= f.logic50 <= xmax):
        side = "below the lowest" if f.logic50 < xmin else "above the highest"
        A.add("IC50_NOT_BRACKETED", "fatal",
              f"Fitted IC50 lies {side} tested concentration ({10**xmin:.3g}-{10**xmax:.3g} {unit}); IC50 is undefined and no value is reported.")
        return A
    A.logic50 = float(f.logic50)
    A.ic50 = float(10 ** f.logic50)

    # plateaus: fraction of the way to the asymptote reached at the dose extremes
    frac_lo = 1 / (1 + 10 ** (f.hill * (xmin - f.logic50)))   # 1 means at y0
    frac_hi = 1 - 1 / (1 + 10 ** (f.hill * (xmax - f.logic50)))  # 1 means at yinf
    if frac_lo < 1 - PLATEAU_FRAC:
        A.add("MISSING_PLATEAU_ZERO_DOSE", "warn", f"Curve has covered only {frac_lo:.0%} of the way back to the zero-dose plateau at the lowest dose; y0 is extrapolated.")
    if frac_hi < 1 - PLATEAU_FRAC:
        A.add("MISSING_PLATEAU_HIGH_DOSE", "warn", f"Curve reaches only {frac_hi:.0%} of the high-dose plateau at the highest dose; yinf is extrapolated.")
    if f.at_bounds:
        A.add("PARAM_AT_BOUND", "warn", f"Parameter(s) at optimizer bound: {', '.join(f.at_bounds)}.")
    if f.hill < HILL_LOW or f.hill > HILL_HIGH:
        A.add("EXTREME_HILL", "warn", f"Hill slope {f.hill:.2f} outside [{HILL_LOW}, {HILL_HIGH}].")
    if A.rmse_over_span > RMSE_SPAN_MAX:
        A.add("POOR_FIT", "warn", f"Residual RMSE is {A.rmse_over_span:.0%} of curve span (> {RMSE_SPAN_MAX:.0%}).")
    if nconc < 7:
        A.add("FEW_CONCENTRATIONS", "warn", f"Only {nconc} distinct concentrations.")

    if do_boot:
        b = bootstrap_ci(x, resp, f, n_boot=n_boot, method=boot_method, seed=seed)
        A.boot = b
        if b["n_ok"] < 0.8 * n_boot:
            A.add("BOOTSTRAP_UNSTABLE", "fatal",
                  f"Only {b['n_ok']}/{n_boot} bootstrap resamples were usable; no interval can be trusted, so no IC50 is reported.")
            A.ic50 = A.logic50 = None
            A.status = "undefined"
            return A
        if b["n_ok"] >= 20:
            lo, hi = b["ci"][2]
            A.logic50_ci = (float(lo), float(hi))
            A.ic50_ci = (float(10 ** lo), float(10 ** hi))
            if hi - lo > CI_WIDTH_LOG_MAX:
                A.add("WIDE_CI", "warn", f"95% CI spans {hi - lo:.2f} log10 units (> {CI_WIDTH_LOG_MAX}).")
            if lo < xmin or hi > xmax:
                A.add("IC50_CI_NOT_BRACKETED", "fatal",
                      f"The 95% bootstrap interval ({10**lo:.3g}-{10**hi:.3g} {unit}) extends beyond the tested range "
                      f"({10**xmin:.3g}-{10**xmax:.3g} {unit}); the IC50 is not reliably bracketed and no value is reported.")
                A.ic50 = A.ic50_ci = A.logic50 = A.logic50_ci = None
                A.status = "undefined"
                return A
    A.status = "warn" if any(s == "warn" for _, s, _ in A.flags) else "ok"
    return A
