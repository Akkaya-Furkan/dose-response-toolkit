# Dose–response toolkit — how reliable is an IC50?

**Question.** How reliable are IC50 estimates from four-parameter logistic fits?

## Data
Public dose–response data (e.g. PubChem BioAssay) and synthetic data with known parameters.

## Approach
1. 4PL fitting with sensible starting values and bounds
2. Parameter recovery on synthetic data
3. Bootstrap confidence intervals
4. QC flags: missing plateaus, extreme Hill slopes, poor fits; PDF/HTML report

## Deliverables
Toolkit, example report, write-up.

## Limitations
- If the dose range does not bracket the IC50, the estimate is undefined; the tool flags it rather than reporting a number.

## Status
Planned — part of a 30-project computational biology and scientific software portfolio.
