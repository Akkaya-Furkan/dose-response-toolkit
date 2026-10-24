"""Synthetic dose-response data with known parameters."""
from __future__ import annotations

import numpy as np
from .model import fourpl


def log_dilution(top_conc, n=10, factor=3.0):
    """Concentrations (descending dilution series, returned ascending)."""
    return np.sort(top_conc / factor ** np.arange(n))


def simulate(conc, y0=100.0, yinf=0.0, ic50=1.0, hill=1.0, noise_sd=5.0, reps=1, seed=0):
    rng = np.random.default_rng(seed)
    c = np.repeat(np.asarray(conc, float), reps)
    y = fourpl(np.log10(c), y0, yinf, np.log10(ic50), hill) + rng.normal(0, noise_sd, len(c))
    return c, y
