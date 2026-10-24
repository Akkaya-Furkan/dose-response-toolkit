"""Bootstrap confidence intervals for 4PL parameters."""
from __future__ import annotations

import numpy as np
from .model import fit_4pl, fourpl


def bootstrap_ci(x, y, fit=None, n_boot=500, method="residual", level=0.95, seed=0):
    """Percentile bootstrap.

    method="case": resample (x, y) pairs with replacement.
    method="residual": resample residuals of the original fit onto fitted values.
    Returns dict with ``samples`` (n_ok x 4 array: y0, yinf, logic50, hill),
    ``ci`` (4 x 2), ``n_ok`` and ``n_boot``.
    """
    if n_boot < 100:
        raise ValueError("n_boot must be >= 100")
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    fit = fit or fit_4pl(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    if method == "residual":
        fitted = fourpl(x, *fit.params)
        res = y - fitted
        res = res * np.sqrt(n / max(n - 4, 1))  # inflate for lost dof
    samples = []
    for _ in range(n_boot):
        if method == "case":
            idx = rng.integers(0, n, n)
            xb, yb = x[idx], y[idx]
            if len(np.unique(xb)) < 5:
                continue
        elif method == "residual":
            xb = x
            yb = fitted + rng.choice(res, n, replace=True)
        else:
            raise ValueError(method)
        f = fit_4pl(xb, yb, p0=fit.params)
        if f.success:
            samples.append(f.params)
    samples = np.array(samples).reshape(-1, 4)
    a = (1 - level) / 2
    if len(samples) >= 20:
        ci = np.quantile(samples, [a, 1 - a], axis=0).T
    else:
        ci = np.full((4, 2), np.nan)
    return {"samples": samples, "ci": ci, "n_ok": len(samples), "n_boot": n_boot,
            "method": method, "level": level}
