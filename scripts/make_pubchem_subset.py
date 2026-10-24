"""Extract a stratified subset of PubChem BioAssay AID 411 (qHTS firefly luciferase inhibitors).

Source file: https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data/0000001_0001000.zip  -> 0000001_0001000/411.csv.gz
Usage: python scripts/make_pubchem_subset.py 411.csv.gz data/pubchem_aid411_subset.csv
Keeps 100 Active, 100 Inconclusive and 100 Inactive substances (seed 14) in long format.
"""
import csv, gzip, random, re, sys

src, dst = sys.argv[1], sys.argv[2]
rows = {"Active": [], "Inconclusive": [], "Inactive": []}
with gzip.open(src, "rt", newline="") as fh:
    rd = csv.DictReader(fh)
    conc_cols = [(c, float(re.match(r"Activity at ([\d.]+) uM", c).group(1)))
                 for c in rd.fieldnames if c.startswith("Activity at")]
    for r in rd:
        if r["PUBCHEM_SID"].isdigit() and r["PUBCHEM_ACTIVITY_OUTCOME"] in rows:
            rows[r["PUBCHEM_ACTIVITY_OUTCOME"]].append(r)
rng = random.Random(14)
with open(dst, "w", newline="") as out:
    w = csv.writer(out)
    w.writerow(["aid", "sid", "outcome", "curve_class", "depositor_logac50_M", "depositor_hill",
                "depositor_r2", "conc_uM", "response_pct"])
    for k, lst in rows.items():
        for r in sorted(rng.sample(lst, 100), key=lambda r: int(r["PUBCHEM_SID"])):
            for col, c in conc_cols:
                if r[col] != "":
                    w.writerow([411, r["PUBCHEM_SID"], k, r["Fit_CurveClass"], r["Fit_LogAC50"],
                                r["Fit_HillSlope"], r["Fit_R2"], c, r[col]])
