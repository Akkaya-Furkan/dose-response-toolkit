"""Reading dose-response tables."""
from __future__ import annotations

import csv
from collections import OrderedDict
import numpy as np


def read_long_csv(path, id_col, conc_col, resp_col):
    """Return OrderedDict id -> (conc array, response array, first-row dict)."""
    out = OrderedDict()
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        missing = [c for c in (id_col, conc_col, resp_col) if c not in (rd.fieldnames or [])]
        if missing:
            raise KeyError(f"column(s) not found in {path}: {missing}; available: {rd.fieldnames}")
        for row in rd:
            try:
                c, r = float(row[conc_col]), float(row[resp_col])
            except (ValueError, TypeError):
                continue
            k = row[id_col]
            if k not in out:
                out[k] = ([], [], row)
            out[k][0].append(c)
            out[k][1].append(r)
    return OrderedDict((k, (np.array(a), np.array(b), row)) for k, (a, b, row) in out.items())
