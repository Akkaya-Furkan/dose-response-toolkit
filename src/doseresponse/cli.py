"""Command line: drtk fit data.csv --id sid --conc conc_uM --resp response_pct --out report.html"""
from __future__ import annotations

import argparse
import json
from .io import read_long_csv
from .qc import analyze
from .report import html_report, pdf_report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="drtk", description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--id", required=True, help="column identifying a curve")
    ap.add_argument("--conc", required=True)
    ap.add_argument("--resp", required=True)
    ap.add_argument("--unit", default="uM")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--method", choices=["case", "residual"], default="residual")
    ap.add_argument("--html")
    ap.add_argument("--pdf")
    ap.add_argument("--json")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    data = read_long_csv(a.csv, a.id, a.conc, a.resp)
    items = []
    for i, (k, (c, r, _)) in enumerate(data.items()):
        if a.limit and i >= a.limit:
            break
        items.append((k, analyze(c, r, unit=a.unit, n_boot=a.boot, boot_method=a.method)))
    if a.html:
        open(a.html, "w").write(html_report(items))
    if a.pdf:
        pdf_report(items, a.pdf)
    out = {str(k): A.to_dict() for k, A in items}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    for k, A in items:
        print(k, A.status, "IC50=" + (f"{A.ic50:.4g}" if A.ic50 is not None else "undefined"), ",".join(c for c, _, _ in A.flags))


if __name__ == "__main__":
    main()
