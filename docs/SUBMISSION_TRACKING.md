# Submission Tracking

This file tracks platform submissions and paper-relevant diagnostics. Keep it limited to the active C03 baseline plus TSViT/C20 comparison evidence.

## Submission 2026-04-30: C03 Full-Data Baseline

### Identity

- Local submission repo: `/Users/yusuf/Desktop/finalround/track1_kybelix`
- Git commit: `c5f95232a4670f5f2b461ada3fff0b966a6a276f`
- Commit message: `Submit C03 full-data baseline`
- GitLab pipeline ID: `3337`
- GitLab build job ID: `8288`
- GitLab create-job ID: `8289`
- Nanhu job ID: `jb-aitrain-155893052943654656`
- Algorithm UUID: `5fcf3423-9b30-48f2-b600-79296f6e6bfd`
- Image tag: `zero2x_repository:c5f95232-3337`
- Image digest: `sha256:653e0e0743b413d4a0e23e2f481ca43a36167f388a39bcffd2706ae648151b72`

### Model

- Candidate: `C03`
- Model type: `query_cnn_transformer`
- Checkpoint path in image: `/workspace/checkpoints/model.pt`
- Checkpoint size: `12.55 MB`
- Checkpoint epoch: `75`
- Checkpoint metric: `fixed_epoch_full_data`
- Parameter count from local diagnostics: `3,275,146`
- Input channels: `24`
- Patch size: `15`
- Crop classes: `3`
- Phenophase classes: `7`
- Uses mask channels: `true`
- Uses auxiliary features: `false`
- Uses crop consistency: `true`
- Stage postprocess: `none`
- Point-stage bijection: `false`

### Platform Environment

- Runner build environment: Docker in Kubernetes
- Runtime compute spec: `Z2120-cuda12.8-80Gx1-8C100G`
- Runtime node: `nm129-a100-80g-113`
- Python: `3.11.10`
- PyTorch: `2.5.1+cu121`
- CUDA available: `true`
- CUDA device count: `1`
- GPU: `NVIDIA A100-SXM4-80GB`
- Docker server CPU count during image build: `16`
- Docker server memory during image build: `46.77 GiB`

### Timing

- Build job window from GitLab timestamps: `1777508637` to `1777508858`, about `221 s`.
- Java service job-creation window: `1777508860` to `1777508867`, about `7 s`.
- Submission runtime start: `2026-04-30T00:27:54Z`
- Submission runtime done: `2026-04-30T00:28:36Z`
- End-to-end container runtime: about `42 s`.
- Package validation start: `08:27:56.445`
- Package validation output: `08:27:57.880`
- Package validation duration: about `1.43 s`.
- Inference command start: `08:27:58.095`
- Inference JSON output: `08:28:36.062`
- Inference command duration: about `37.97 s`.
- Result JSON write confirmed: `08:28:36.490`

Current limitation: this log does not split preprocessing, model forward, and JSON serialization time. Add finer timing instrumentation before the next diagnostic submission if we need exact latency attribution.

### Input And Output

- Platform points file: `/input/test_point.csv`
- TIFF directory: `/input/region_test`
- TIFF count from `run.sh`: `2235`
- Query rows: `942`
- Unique output keys / final result rows: `930`
- Duplicate output-key rows: `12`
- Duplicate output-key conflicts: `0`
- Unique points extracted: `171`
- Samples kept: `171`
- Samples dropped: `0`
- Output JSON: `/mnt/si000886fq1w/default/output/result.json`

### Patch Dataset Diagnostics

- Patch tensor shape: `[171, 27, 12, 15, 15]`
- Max timesteps: `27`
- Total time cells: `2997`
- Padded time cells: `1620`
- Samples with time padding: `164`
- Missing band cells: `0`
- Samples with missing bands: `0`
- Observed patch pixels: `8,091,900`
- Valid patch pixels: `7,485,120`
- Invalid patch pixels: `606,780`
- Global valid pixel ratio: `0.9250139028`
- Samples with any invalid pixel: `165`
- Samples requiring edge replication: `7`
- Edge replication band cells: `1632`
- Center clamping band cells: `0`
- File read failures: `0`
- Patch extraction errors: `0`
- Mapping status counts: `unique_region=169`, `overlap_multi_region=2`

### Prediction Distribution

Crop prediction counts over `942` query rows:

- `corn`: `474`
- `rice`: `196`
- `soybean`: `272`

Stage prediction counts over `942` query rows:

- `Greenup`: `47`
- `MidGreenup`: `147`
- `Peak`: `151`
- `Maturity`: `136`
- `MidSenescence`: `164`
- `Senescence`: `239`
- `Dormancy`: `58`

Sample output rows from log:

- `124.703696_48.543523_2018/9/1 -> [corn, Senescence]`
- `125.331726_48.768485_2018/6/21 -> [soybean, MidGreenup]`
- `125.331726_48.768485_2018/7/31 -> [soybean, Peak]`

### Scores

- Hidden/test score: `0.994132`.
- Visible secondary metric from platform table: `0.938218`.
- Card consumption time: `0.0119 H`.
- Platform table submission time: `2026-04-30 08:27:55`.
- Hidden/test crop macro-F1: not safely confirmed from the pasted table column labels.
- Hidden/test rice-stage macro-F1: not safely confirmed from the pasted table column labels.
- User note: this C03 submission performed slightly better than the previous submitted model.

Important: if `0.938218` were crop macro-F1 under the documented formula, the implied rice-stage macro-F1 would be above `1.0`, which is impossible. Therefore the table has at least one ambiguous column label, or `0.938218` is not the crop macro-F1. Use a platform screenshot/export before citing component metrics in the paper.

## Current Interpretation

C03 is validated as the active operational baseline:

- It ran successfully on the platform.
- It used GPU automatically.
- It produced the required `/output/result.json`.
- It had no duplicate output-key conflicts.
- It had no file read or patch extraction failures.
- It is much smaller and faster than TSViT in local proxy benchmarks.

TSViT remains the comparison architecture for the paper, not the active submission model.

## Submission 2026-04-30: C20 Fourier + Stage Postprocess

### Identity

- GitLab checkout commit in log: `864547fe`
- GitLab pipeline ID: `3409`
- GitLab build job ID: `8458`
- GitLab create-job ID: `8459`
- Nanhu job ID: `jb-aitrain-155912915927987904`
- Algorithm UUID: `2b089539-87f9-4502-b810-a8f2ce2e3141`
- Image tag: `zero2x_repository:864547fe-3409`
- Image digest: `sha256:6db318f21235a6adc7d9462849f382f1cce83150922b01f59de8391b4d10aba1`
- Platform table submission time: `2026-04-30 18:59:18`
- Status: `Succeed`

Important: this was not the P3 optimized C03 commit. The log shows checkout `864547fe`, C20/Fourier checkpoint epoch `77`, and runtime config with `stage_postprocess: transition_viterbi`.

### Model

- Candidate represented by checkpoint: `C20 Fourier`
- Model type: `query_cnn_transformer`
- Checkpoint path in image: `/workspace/checkpoints/model.pt`
- Checkpoint size: `12.57 MB`
- Checkpoint epoch: `77`
- Checkpoint metric: `fixed_epoch_full_data`
- Input channels: `24`
- Patch size: `15`
- Crop classes: `3`
- Phenophase classes: `7`
- Time encoding: `fourier`
- Query encoding: `fourier`
- Time encoding harmonics: `6`
- Uses mask channels: `true`
- Uses auxiliary features: `false`
- Uses crop consistency: `true`
- Stage postprocess: `transition_viterbi`
- Point-stage bijection: `false`

### Platform Environment

- Runtime node: `nm129-a100-80g-113`
- Python: `3.11.10`
- PyTorch: `2.5.1+cu121`
- CUDA available: `true`
- CUDA device count: `1`
- GPU: `NVIDIA A100-SXM4-80GB`
- Docker server CPU count during image build: `16`
- Docker server memory during image build: `46.77 GiB`

### Timing

- Build job window from GitLab timestamps: `1777546512` to `1777546743`, about `231 s`.
- Java service job-creation window: `1777546745` to `1777546752`, about `7 s`.
- Runtime start: `2026-04-30T10:59:18Z`
- Runtime done: `2026-04-30T10:59:55Z`
- End-to-end container runtime from log: about `36.90 s`.
- Card consumption time from platform table: `0.0106 H`, about `38.16 s`.
- Package validation start: `18:59:20.017`
- Package validation output: `18:59:21.487`
- Package validation duration: about `1.47 s`.
- Inference command start: `18:59:21.843`
- Inference JSON output: `18:59:54.551`
- Inference command duration to JSON report: about `32.71 s`.
- Result JSON write confirmed: `18:59:55.003`
- Inference command duration to write confirmation: about `33.16 s`.

Fine-grained timing limitation: although this run was intended to include time metrics, the printed inference JSON in the pasted log does not include `timing_seconds`. Therefore we can use the outer log timestamps above, but not a split of patch extraction, model load, forward pass, postprocess, or JSON write.

### Input And Output

- Platform points file: `/input/test_point.csv`
- TIFF directory: `/input/region_test`
- TIFF count from `run.sh`: `2235`
- Query rows: `942`
- Unique output keys / final result rows: `930`
- Duplicate output-key rows: `12`
- Duplicate output-key conflicts: `0`
- Unique points extracted: `171`
- Samples kept: `171`
- Samples dropped: `0`
- Output JSON: `/mnt/si000886fq1w/default/output/result.json`

### Patch Dataset Diagnostics

- Patch tensor shape: `[171, 27, 12, 15, 15]`
- Max timesteps: `27`
- Total time cells: `2997`
- Padded time cells: `1620`
- Samples with time padding: `164`
- Missing band cells: `0`
- Samples with missing bands: `0`
- Observed patch pixels: `8,091,900`
- Valid patch pixels: `7,485,120`
- Invalid patch pixels: `606,780`
- Global valid pixel ratio: `0.9250139028`
- Samples with any invalid pixel: `165`
- Samples requiring edge replication: `7`
- Edge replication band cells: `1632`
- Center clamping band cells: `0`
- File read failures: `0`
- Patch extraction errors: `0`
- Mapping status counts: `unique_region=169`, `overlap_multi_region=2`

### Prediction Distribution

Crop prediction counts over `942` query rows:

- `corn`: `468`
- `rice`: `196`
- `soybean`: `278`

Stage prediction counts over `942` query rows:

- `Greenup`: `47`
- `MidGreenup`: `155`
- `Peak`: `145`
- `Maturity`: `159`
- `MidSenescence`: `224`
- `Senescence`: `149`
- `Dormancy`: `63`

Sample output rows from log:

- `124.703696_48.543523_2018/9/1 -> [corn, Senescence]`
- `125.331726_48.768485_2018/6/21 -> [soybean, MidGreenup]`
- `125.331726_48.768485_2018/7/31 -> [soybean, Peak]`

### Scores

- Hidden/test score: `0.907497`
- Card consumption time: `0.0106 H`
- Visible component metrics: not present in pasted platform table/log.

### Interpretation

- C20 Fourier is faster by card time than the C03 full-data baseline: `0.0106 H` vs `0.0119 H`, about `0.0013 H` or `4.68 s` lower.
- C20 Fourier is much worse in hidden score than C03: `0.907497` vs `0.994132`, a drop of `0.086635`.
- This confirms the local robustness concern: despite very strong normal validation loss, C20 is not currently safe as the operational submission model.
- C20 remains useful as an innovation and ablation story for Fourier temporal encoding, but not as the final candidate unless we can explain and fix the hidden-test degradation.

## Submission 2026-04-30: C03 P3 Optimized Pipeline

### Identity

- Local submission repo: `/Users/yusuf/Desktop/finalround/track1_kybelix`
- Git commit: `5d87087888dcc537fd6dc9b897cb5f37e203fc1f`
- Commit message: `Submit C03 P3 optimized pipeline`
- Remote branch: `origin/main`
- Remote verification: `refs/heads/main` points to `5d87087888dcc537fd6dc9b897cb5f37e203fc1f`
- Status: pushed; awaiting GitLab/platform result.

### Model

- Candidate: `C03`
- Checkpoint: unchanged C03 epoch-75 full-data model, copied as `/workspace/checkpoints/model.pt`.
- Checkpoint size: `12.55 MB`
- Model type: `query_cnn_transformer`
- Pipeline version: `P3`

### P3 Pipeline Changes

- `fast_inventory: true`
- `use_gdal_env: true`
- `use_in_memory_patches: true`
- `write_patch_npz: false`
- `compressed_patch_npz: false`
- `use_crop_consistency: true`
- Removed explicit unused `use_point_stage_bijection` and `stage_postprocess` fields from final config; code defaults remain `false` and `none`.

### Local Validation Before Push

- Package validation in submission repo: passed.
- Direct submission-repo smoke on local platform-shaped input: passed.
- Smoke queries: `21`
- Unique output keys: `21`
- Duplicate output-key conflicts: `0`
- Missing band cells: `0`
- File read failures: `0`
- Patch extraction errors: `0`
- P3 flags confirmed active in smoke report: `fast_inventory=true`, `use_in_memory_patches=true`, `use_gdal_env=true`.

### Platform Metrics

- Nanhu job ID: `jb-aitrain-155924679797274304`
- GitLab pipeline ID: `3448`
- GitLab build job ID: `8541`
- GitLab create-job ID: `8542`
- Algorithm UUID: `97ea585c-1584-4bc4-bda1-02ddc5b6670c`
- Image tag: `zero2x_repository:5d870878-3448`
- Image digest: `sha256:75fbe05c30bc644bb5f7a2c97d6abe176e3664a72ad662912be662949bc866f6`
- Hidden/test score: not pasted yet.
- Visible secondary metric: not pasted yet.
- Card consumption time: user observed no meaningful change from C03 baseline.
- Runtime diagnostics: available; see below.

When the platform row appears, add exact score, visible metrics, card consumption time, timestamp, job ID, and whether result rows/duplicate conflicts match expectations.

### Platform Runtime Result

The P3 config was active in the platform run:

- Checkout commit: `5d870878`
- Runtime config included `fast_inventory: true`
- Runtime config included `use_gdal_env: true`
- Runtime config included `use_in_memory_patches: true`
- Runtime config included `write_patch_npz: false`
- Runtime config omitted explicit stage postprocess; runtime used default `stage_postprocess: none`
- Runtime used `cuda` on `NVIDIA A100-SXM4-80GB`

Outer timing:

- Runtime start: `2026-04-30T17:13:16Z`
- Runtime done: `2026-04-30T17:13:58Z`
- End-to-end container runtime from log: about `41.35 s`
- Package validation start: `01:13:18.572`
- Package validation output: `01:13:20.180`
- Package validation duration: about `1.61 s`
- Inference command start: `01:13:20.397`
- Inference JSON output: `01:13:57.455`
- Inference command duration to JSON report: about `37.06 s`
- Result JSON write confirmed: `01:13:57.991`
- Inference command duration to write confirmation: about `37.59 s`

Internal timing from P3 `timing_seconds`:

- `total_seconds`: `35.3711`
- `patch_extraction_seconds`: `33.4366`
- `model_load_seconds`: `0.4714`
- `npz_and_query_rows_seconds`: `0.2252`
- `batch_prepare_seconds`: `0.6330`
- `model_forward_seconds`: `0.4281`
- `logits_to_cpu_seconds`: `0.0011`
- `postprocess_seconds`: `0.0340`
- `write_result_seconds`: `0.0354`
- `queries_per_total_second`: `26.6319`
- `queries_per_forward_second`: `2200.5350`

Internal patch-build timing:

- `load_points_seconds`: `0.0106`
- `audit_tiff_files_seconds`: `0.1546`
- `select_file_index_seconds`: `0.0`
- `build_region_catalog_seconds`: `0.0424`
- `map_points_to_regions_seconds`: `0.0142`
- `prepare_metadata_seconds`: `0.0006`
- `build_lookup_and_allocate_seconds`: `0.0231`
- `extract_raster_patches_seconds`: `33.1392`
- `summarize_metadata_seconds`: `0.0174`
- `write_npz_seconds`: `0.0`
- `build_report_seconds`: `0.0264`
- `total_build_patch_dataset_seconds`: `33.4304`

P3 output checks:

- Query rows: `942`
- Unique output keys / result rows: `930`
- Duplicate output-key rows: `12`
- Duplicate output-key conflicts: `0`
- Unique points extracted: `171`
- Samples kept: `171`
- Samples dropped: `0`
- Missing band cells: `0`
- File read failures: `0`
- Patch extraction errors: `0`
- Mapping status counts: `unique_region=169`, `overlap_multi_region=2`

P3 prediction distribution:

- Crop counts: `corn=474`, `rice=196`, `soybean=272`
- Stage counts: `Greenup=47`, `MidGreenup=147`, `Peak=151`, `Maturity=136`, `MidSenescence=164`, `Senescence=239`, `Dormancy=58`

Interpretation:

- P3 worked correctly and activated all intended config flags.
- Card time did not visibly improve because the remaining bottleneck is raster patch extraction: `33.1392 s` out of `35.3711 s` internal runtime.
- P3 removed the selected-file-index and NPZ-write costs, but those are tiny on platform compared with reading `35,964` band/time patch windows from TIFFs.
- The model itself is not the runtime bottleneck: forward pass was only `0.4281 s`.
- Relative to the earlier C03 baseline log, outer inference duration improved only slightly: about `37.97 s` previously vs about `37.59 s` to write confirmation here. This is too small to reliably change a coarse card-consumption display.

## F1 Improvement Notes

Without hidden labels, direct F1 tuning is not observable. We can still try to improve expected F1, but every change must be justified by validation, diagnostics, or robustness tests.

Possible C03-focused improvement directions:

- Better latency diagnostics before changing model behavior.
- Data-prep optimization before model compression, because local smoke timing shows patch extraction dominates model forward time.
- Band importance analysis before band or channel pruning.
- Dynamic quantization and model quantization after the frozen baseline/working duplicate comparison is stable.
- Teacher-student distillation from submitted C03 to a smaller C03-style model; only use C20/TSViT as teachers if they show better robustness or calibration.
- Safer postprocessing only if it is supported by validation evidence.
- C03 distillation/compression or quantization for speed and deployment story.
- C03 architecture refinements with the same input/output contract.
- C03 robustness tests around query-date shifts, missing bands, valid-mask behavior, and duplicate-key handling.

Avoid making hidden-test-driven edits from a single score unless we can explain the mechanism and verify it locally.
