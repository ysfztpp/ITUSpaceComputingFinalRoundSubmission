# Sanity Checks

This bundle is intentionally limited to two candidates:

- `C03`: primary full-data model, `query_cnn_transformer`.
- `TSViT`: comparison model, `query_tsvit`.

The shared `models/temporal_transformer.py` file is not a third candidate. It is retained because both final models import its time-encoding helpers.

## PDF-Derived Requirements

- Platform inference must write `/output/result.json`.
- Platform input is inference-only, with point CSV plus `/input/region_test/*.tiff`.
- Score is `0.4 * crop_macro_f1 + 0.6 * rice_stage_macro_f1`.
- Rice stage scoring is a strict double hit: rice crop and stage must both match.

## Current Metrics Inventory

| Candidate | Evidence | Epoch | Score | Crop F1 | Rice Stage F1 | Loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C03 | validation archive | 75 | 1.000000 | 1.000000 | 1.000000 | 0.061087 |
| C03 full-data | train-only final checkpoint | 75 | 0.999066 | 1.000000 | 0.998443 | 0.057580 |
| TSViT full-data | train-only final checkpoint | 42 | 1.000000 | 1.000000 | 1.000000 | 0.166929 |

The local archive does not currently include a separate TSViT validation metrics sidecar. Treat TSViT as an architecture and innovation comparison until a validation or hidden-platform metric is available.

## Run

```bash
.venv/bin/python scripts/run_sanity_checks.py
```

Use this faster form if PyTorch checkpoint loading is not available in the current environment:

```bash
.venv/bin/python scripts/run_sanity_checks.py --skip-checkpoint-validation
```

## Benchmark

Synthetic forward timing for both candidates:

```bash
.venv/bin/python scripts/benchmark_models.py --candidate all
```

End-to-end timing on a platform-shaped input folder:

```bash
.venv/bin/python scripts/benchmark_models.py --candidate c03 --input-root /path/to/input
```
