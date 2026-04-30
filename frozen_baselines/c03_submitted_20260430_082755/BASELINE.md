# C03 Submitted Baseline

This folder freezes the C03 model package that produced the latest successful platform submission.

## Platform Result

- Job ID: `jb-aitrain-155893052943654656`
- Track: `track1`
- Submission time: `2026-04-30 08:27:55`
- Status: `Succeed`
- Score: `0.994132`
- Crop macro-F1: `0.938218`
- Card consumption time: `0.0119 H`

The platform row did not expose the rice phenology/stage macro-F1 in the pasted table. From the competition formula, the implied rice-stage macro-F1 is approximately `1.0314086667`, which is impossible for an F1 score. Therefore the second visible metric is not sufficient to recover the official component scores, or the table column labels differ from our assumption. Keep exact component naming tied to a screenshot/export before using it in the paper.

## Frozen Files

- `c03_submitted_model.pt`
- `submission_c03_submitted.json`
- `train_full_data_c03_epoch75_submitted.json`

## Source Checksums

- `checkpoints/c03_full_data_model.pt`: `8691eb15fd1a591ca234d94a6cb9e0da9d178b7ce5bc109a6b69b9d126de8954`
- `configs/submission_c03.json`: `030e615d85b20c65f35531c2534cb6a969fcd4c7cf8ec596fb095db892875e23`
- `configs/train_full_data_c03_epoch75.json`: `571adc3016affc1a0b4b76dd22db28e6948163aec6d532b9b46f8df020077cf3`
- `artifacts/normalization/train_patch_band_stats.json`: `7185e525c92578817a93d27285db010f077697efd27990331210beb098a3f9dd`

## Rule

Do not edit these frozen files. Optimization, quantization, pruning, distillation, and data-prep changes should target `checkpoints/optimization/c03_working_model.pt` and configs under `configs/optimization/`.
