"""4PL model and least-squares fit.

Parametrisation (x = log10 concentration):

    y(x) = yinf + (y0 - yinf) / (1 + 10**(hill * (x - logic50)))

``y0`` is the plateau at zero dose, ``yinf`` the plateau at infinite dose,
``hill`` is a positive slope magnitude; the direction of the curve is given by
the sign of ``y0 - yinf``. This keeps the optimisation well conditioned for both
inhibition and activation curves.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

HILL_BOUNDS = (0.1, 10.0)


def fourpl(x, y0, yinf, logic50, hill):
    x = np.asarray(x, dtype=float)
    z = np.clip(hill * (x - logic50), -300, 300)
    return yinf + (y0 - yinf) / (1.0 + 10.0 ** z)


@dataclass
class FitResult:
    y0: float
    yinf: float
    logic50: float
    hill: float
    sse: float
    n: int
    success: bool
    at_bounds: tuple = ()

    @property
    def params(self):
        return np.array([self.y0, self.yinf, self.logic50, self.hill])

    @property
    def rmse(self):
        return float(np.sqrt(self.sse / max(self.n - 4, 1)))


def _starts(x, y):
    xs = np.unique(x)
    lo = np.median(y[x == xs[0]])
    hi = np.median(y[x == xs[-1]])
    mid = 0.5 * (lo + hi)
    # interpolate where the mean curve crosses the midpoint
    ym = np.array([y[x == v].mean() for v in xs])
    cross = []
    for i in range(len(xs) - 1):
        a, b = ym[i] - mid, ym[i + 1] - mid
        if a == 0:
            cross.append(xs[i])
        elif a * b < 0:
            cross.append(xs[i] + (xs[i + 1] - xs[i]) * a / (a - b))
    centres = cross[:1] + [float(np.median(xs))] + list(np.quantile(xs, [0.25, 0.75]))
    out = []
    for c in centres:
        for h in (0.7, 1.0, 2.0):
            out.append([lo, hi, c, h])
    return out


def fit_4pl(x, y, p0=None, extra_starts=True) -> FitResult:
    """Fit the 4PL to log10-concentration ``x`` and response ``y``.

    The response is rescaled to [0, 1] internally so tolerances do not depend on
    units. ``p0`` (optional, original response units) is a warm start, e.g. in
    bootstrap loops; with ``extra_starts`` two default starts are tried as well so
    a warm start cannot lock the fit into a worse local minimum. Without ``p0``
    a multi-start search is run.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    lo_y, span = float(y.min()), float(np.ptp(y))
    if not np.isfinite(span) or span <= 0:
        return FitResult(*([np.nan] * 4), np.nan, len(x), False)
    yn = (y - lo_y) / span
    lb = [-10.0, -10.0, x.min() - 2.0, HILL_BOUNDS[0]]
    ub = [11.0, 11.0, x.max() + 2.0, HILL_BOUNDS[1]]
    starts = list(_starts(x, yn))
    if p0 is not None:
        w = np.array([(p0[0] - lo_y) / span, (p0[1] - lo_y) / span, p0[2], p0[3]])
        starts = [w] + (starts[1:2] + starts[4:5] if extra_starts else [])
    best = None
    for q in starts:
        q = np.clip(q, np.array(lb) + 1e-6, np.array(ub) - 1e-6)
        try:
            r = least_squares(lambda t: fourpl(x, *t) - yn, q, bounds=(lb, ub))
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return FitResult(*([np.nan] * 4), np.nan, len(x), False)
    t = best.x
    names = ("y0", "yinf", "logic50", "hill")
    at = tuple(n for n, v, a, b in zip(names, t, lb, ub) if abs(v - a) < 1e-4 or abs(v - b) < 1e-4)
    p = np.array([t[0] * span + lo_y, t[1] * span + lo_y, t[2], t[3]])
    return FitResult(*p, float(2 * best.cost) * span ** 2, len(x), bool(best.success), at)
