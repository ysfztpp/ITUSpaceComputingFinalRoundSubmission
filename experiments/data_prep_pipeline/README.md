# Data-Prep Pipeline Experiments

This area is for numbered C03 inference pipeline experiments.

## Versions

- `P0`: frozen submitted C03 reference files.
- `P1`: faster selected-file index implementation. This is a code-level optimization in `preprocessing/inventory.py`.
- `P2`: P1 plus opt-in GDAL open/cache settings and in-memory patch arrays.

## Fairness Rule

Each benchmark run writes into a new timestamped directory under `artifacts/experiments/data_prep_pipeline/`. The script deletes only that newly created run directory if it already exists, then creates fresh `work/` and `output/` folders per variant. This prevents stale temporary NPZs or stale result JSONs from affecting measurements.

The script does not clear the operating-system file cache because that generally requires administrator privileges and can affect the whole machine. For paper reporting, use repeated rounds and report median timings.
