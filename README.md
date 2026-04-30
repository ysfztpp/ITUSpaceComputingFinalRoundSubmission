# Project Organized

This folder is a cleaned final-phase bundle built around only two candidate models:

- `C03`: the safest full-data submission baseline.
- `C29 TSViT`: the stronger and heavier transformer candidate kept for comparison and final-phase presentation.

Everything unrelated to these two paths has been removed from this bundle.

## Folder Layout

- `checkpoints/`
  - `c03_full_data_model.pt`
  - `c29_tsvit_model.pt`
- `configs/`
  - `preprocess.json`
  - `train_full_data_c03_epoch75.json`
  - `train_full_data_c29_tsvit_relative_aux_crop_submission.json`
  - `submission_c03.json`
  - `submission_tsvit.json`
- `data/`, `models/`, `preprocessing/`, `training/`
  Minimal code needed for preprocessing, training, validation, and inference.
- `scripts/`
  Entry points for preprocessing, training, submission preparation, and validation.
- `model_cards/`
  Exact archived metadata for the two selected checkpoints.
- `docs/`
  Final-phase summary material aligned to the judging rubric.

## Candidate Summary

| Candidate | Model Type | Checkpoint Size | Params | Fixed Epoch | Main Use |
| --- | --- | ---: | ---: | ---: | --- |
| `C03` | `query_cnn_transformer` | `12.55 MB` | `3.28 M` | `75` | Primary final submission baseline |
| `C29 TSViT` | `query_tsvit` | `88.51 MB` | `23.17 M` | `42` | Advanced comparison model for final phase |

Recommendation:

- Use `C03` as the operational baseline when low risk, small uplink size, and CPU feasibility matter most.
- Use `C29 TSViT` as the "stronger but heavier" architecture for technical comparison, future roadmap, and on-orbit acceleration discussion.

## Quick Commands

Validate the C03 submission package:

```bash
python scripts/validate_submission.py --config configs/submission_c03.json
```

Validate the TSViT submission package:

```bash
python scripts/validate_submission.py --config configs/submission_tsvit.json
```

Prepare a GitLab submission repo with the C03 candidate:

```bash
python scripts/prepare_finalist_submission.py --candidate c03 --submission-repo /path/to/track1_kybelix
```

Prepare a GitLab submission repo with the TSViT candidate:

```bash
python scripts/prepare_finalist_submission.py --candidate tsvit --submission-repo /path/to/track1_kybelix
```

Rebuild the training NPZ from raw TIFF data:

```bash
python scripts/preprocess.py --config configs/preprocess.json
```

Retrain C03 on all labeled data:

```bash
python scripts/train_full_data.py --config configs/train_full_data_c03_epoch75.json
```

Retrain C29 TSViT on all labeled data:

```bash
python scripts/train_full_data.py --config configs/train_full_data_c29_tsvit_relative_aux_crop_submission.json
```

Run platform-style inference:

```bash
INPUT_ROOT=/path/to/input OUTPUT_DIR=/path/to/output ./run.sh
```

Select TSViT explicitly:

```bash
CANDIDATE=tsvit INPUT_ROOT=/path/to/input OUTPUT_DIR=/path/to/output ./run.sh
```

Run the focused sanity checks:

```bash
.venv/bin/python scripts/run_sanity_checks.py
```

Run local model timing benchmarks:

```bash
.venv/bin/python scripts/benchmark_models.py --candidate all
```

## Notes

- The large training NPZ is intentionally not copied into this bundle. Rebuild it with preprocessing or mount it externally.
- `artifacts/normalization/train_patch_band_stats.json` is included because submission validation and inference require it.
- The parent `track1_kybelix` folder is the actual GitLab submission repo. Use this folder for diagnostics and model selection, then copy exactly one candidate into `track1_kybelix` with `scripts/prepare_finalist_submission.py` when ready.
- The presentation-facing explanation is in [docs/FINAL_PHASE_SCORING_BRIEF.md](docs/FINAL_PHASE_SCORING_BRIEF.md).
