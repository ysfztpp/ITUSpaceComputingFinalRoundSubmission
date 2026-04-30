# P1-P4 Data-Prep Optimization Steps

Date: 2026-04-30

## Purpose

This file records the data-prep inference optimization work separately from the model experiments. C03 submitted baseline is frozen; pipeline variants are numbered so we can roll back if a version breaks.

## Versioning Rule

Use numbered versions for future pipeline/model changes:

- `P0`: frozen submitted baseline reference.
- `P1`: first safe data-prep implementation optimization.
- `P2`: second opt-in inference deployment optimization.
- `P3`: final current pre-model optimization; P2 plus inference-only fast TIFF inventory.
- `P4`: P3 plus raster-window batching for points sharing the same raster file.
- Future changes should create `P5`, `P6`, etc. configs/docs/artifacts instead of overwriting the latest working version.

Current version files:

- `configs/optimization/versions/P0_submission_c03_frozen_reference.json`
- `configs/optimization/versions/P1_submission_c03_file_index_fast.json`
- `configs/optimization/versions/P2_submission_c03_gdal_inmemory.json`
- `configs/optimization/versions/P3_submission_c03_fast_inventory_gdal_inmemory.json`
- `configs/submission.json` in the submission repo currently carries the P4 runtime flags.
- `scripts/benchmark_data_prep_p1.py`
- `experiments/data_prep_pipeline/README.md`

## What We Are Testing On

Local benchmark input:

- Input root: `/tmp/kybelix_pipeline_smoke/input`
- Points file: `/tmp/kybelix_pipeline_smoke/input/test_point.csv`
- TIFF folder: `/tmp/kybelix_pipeline_smoke/input/region_test`
- Query rows: `21`
- Unique points: `3`
- Region-test entries in local smoke folder: about `10237`
- Device used for benchmark: CPU
- Model: C03 query-conditioned CNN + temporal transformer
- Checkpoint size: `12.55 MB`
- Patch size: `15 x 15`
- Bands: `12` Sentinel-2 bands plus valid-mask channels in the model input

This is not a hidden-test quality measurement. It is a pipeline correctness and timing smoke. It is useful for paper discussion as a controlled local latency diagnostic, not as a final platform-runtime claim.

Platform context from submitted C03:

- Job ID: `jb-aitrain-155893052943654656`
- Submission time: `2026-04-30 08:27:55`
- Score: `0.994132`
- Card consumption time: `0.0119 H`
- Platform query rows: `942`
- Unique output keys: `930`
- Unique points extracted: `171`
- TIFF count from run log: `2235`
- GPU: `NVIDIA A100-SXM4-80GB`

## How The Inference Data Pipeline Works

The model does not read the raw CSV and TIFF files directly. Before the C03 neural network can predict crop and phenology stage, the inference code has to turn the platform files into model tensors.

Step-by-step:

1. Read `test_point.csv`.
   - This file contains query rows: longitude, latitude, point ID, and query date.
   - Several query dates can belong to the same physical point.

2. Find unique spatial points.
   - If one point has six dates, we only need to extract its satellite patch once.
   - The model later reuses that same patch time series for each query date.

3. Scan `region_test/*.tiff`.
   - Each TIFF file is one Sentinel-2 band for one region and one acquisition date.
   - The filename tells us the region, date, processing level, and band.
   - Example logic: choose band `B08` for `regionX` on a specific date.

4. Select the best TIFF per `(region, date, band)`.
   - Sometimes there can be multiple candidate files for the same logical band/date.
   - The rule is: prefer `L2A`, then stable path order.
   - This creates the selected-file index.

5. Build the region catalog.
   - For each region, open one raster file and read metadata: geographic bounds, image width/height, pixel size.
   - This tells us whether a point is inside a region and how longitude/latitude maps to pixel row/column.

6. Map each point to a region.
   - If a point is inside one region, use that region.
   - If it overlaps multiple regions, choose the best one by time steps, band count, border margin, and region ID.
   - In inference mode, nearest fallback is allowed if needed.

7. Extract small raster windows.
   - For each selected point, date, and band, read a `15 x 15` window around the point.
   - This is much cheaper than reading the whole TIFF.
   - If the point is close to an image edge, the patch is padded by edge replication.

8. Clean invalid pixels.
   - Values must be finite and inside the configured reflectance range.
   - Invalid pixels are filled with `0.0`.
   - A valid-pixel mask is saved so the model knows which pixels were real.

9. Build the model input tensor.
   - Shape conceptually becomes: `points x time x bands x patch_height x patch_width`.
   - C03 uses the 12 reflectance bands plus 12 mask channels, so model input has `24` channels.

10. Run the neural network for each query date.
    - The same spatial patch series can be paired with different query dates.
    - The query date tells the model which phenology stage to predict.

11. Postprocess predictions.
    - Crop consistency forces one crop label per physical point.
    - Duplicate JSON keys are resolved deterministically.

12. Write `/output/result.json`.
    - This is the platform-required output file.

## How The Optimizations Work

### P1: Fast Selected-File Index

Before P1:

- The code used pandas groupby/sort logic to group all valid TIFF rows by `(region, date, band)`.
- This was correct, but slow on the local smoke folder because there are thousands of TIFF entries.
- The slow part was not reading image pixels; it was organizing filenames into groups.

After P1:

- The code uses a normal Python dictionary.
- The dictionary key is `(region_id, date, band_id)`.
- Each TIFF candidate is appended into the matching bucket once.
- Then each bucket is sorted only enough to choose the best candidate.

Why predictions do not change:

- The selection rule is unchanged: prefer `L2A`, then path order.
- The same selected TIFF should be used for each `(region, date, band)`.
- We verified this indirectly by confirming the final output JSON is identical and patch diagnostics are unchanged.

Why it is faster:

- It avoids expensive dataframe groupby/sort overhead.
- The operation is closer to the actual task: build buckets and choose one best file per bucket.

### P2: GDAL Settings And In-Memory Patches

Before P2:

- Rasterio/GDAL opens many TIFF files.
- When opening a file, GDAL may inspect sibling files in the same folder.
- The baseline also writes extracted patches into a compressed temporary NPZ file, then immediately reads that NPZ back for inference.

After P2:

- GDAL is told not to scan the large sibling directory on each file open.
- GDAL cache settings are enabled for local repeated reads.
- The extracted patch arrays stay in memory and go directly into model inference.
- The temporary NPZ write/read round trip is skipped.

Why predictions do not change:

- The same TIFF files are selected.
- The same raster windows are read.
- The same pixel cleaning and masks are used.
- Only the storage/transfer path changes: memory instead of temporary compressed NPZ.

Why it is faster:

- Less directory-scanning overhead during raster opens.
- Less disk I/O and compression/decompression overhead.

### P3: Fast Inference Inventory

Before P3:

- Even after P1, the pipeline still built a general-purpose TIFF audit dataframe.
- That full audit is useful for training/reporting, but submission inference only needs two things:
  - selected file paths for model tensors
  - status counts for sanity reporting

After P3:

- The inference path scans filenames once and directly builds the selected-file index.
- It keeps the full audit path available for non-inference/reporting workflows.
- P3 is enabled with `fast_inventory: true`.

Why predictions do not change:

- The same duplicate handling is used.
- The same supported-band filter is used.
- The same selection rule is used: prefer `L2A`, then path order.
- P3 was tested against P0 and produced identical result JSON.

Why it is faster:

- It skips building a full dataframe with report-only columns during inference.
- It removes the separate selected-file-index phase; in P3 that timing is `0.0 s` because selection happens inside the inventory scan.

## Delay-Time Definitions

- `total_seconds`: full inference call, from entering `run_inference` to written result JSON.
- `patch_extraction_seconds`: the full data-prep stage: point CSV load, TIFF audit, file selection, region mapping, raster window reads, patch cleaning, optional NPZ write.
- `model_load_seconds`: checkpoint load and model construction.
- `npz_and_query_rows_seconds`: normalizer load, patch array load or in-memory handoff, and query-row construction.
- `batch_prepare_seconds`: normalization and tensor creation for model batches.
- `model_forward_seconds`: PyTorch model forward time only.
- `postprocess_seconds`: crop consistency, stage decoding, duplicate-key consistency.
- `write_result_seconds`: JSON serialization to result file.
- `audit_tiff_files_seconds`: scan TIFF names and parse metadata from filenames.
- `select_file_index_seconds`: choose one best file per `(region, date, band)`.
- `build_region_catalog_seconds`: read one raster metadata file per region and build region bounds.
- `extract_raster_patches_seconds`: rasterio window reads and pixel cleaning for patches.
- `write_npz_seconds`: temporary patch NPZ serialization. P2 disables this during inference.

## P0: Frozen Submitted Reference

Role:

- Keep the submitted C03 checkpoint/config recoverable.
- Provide output reference for all pipeline experiments.

Files:

- `frozen_baselines/c03_submitted_20260430_082755/`
- `configs/optimization/versions/P0_submission_c03_frozen_reference.json`

Important caveat:

- If P0 is rerun inside the current codebase, it benefits from safe global code optimizations such as the P1 file-index rewrite. The original submitted container runtime remains documented in `docs/SUBMISSION_TRACKING.md`.

Historical local pre-P1 timing from smoke:

- Total: about `2.97 s`
- Patch extraction: about `2.89 s`
- Selected-file index: about `2.1 s`

## P1: Fast Selected-File Index

Problem:

- `select_file_index` was the dominant local bottleneck.
- It used pandas groupby/sort operations to choose the best path per `(region_id, start_norm, band_id)`.

Change:

- Replaced the groupby-heavy path with a single-pass Python dictionary keyed by `(region_id, start_norm, band_id)`.
- Preserved the same selected-path rule: prefer `L2A`, then path order.
- Preserved reporting fields: candidate count, candidate levels, candidate paths, selected level, selected path.

Latest smoke result after P1:

- Outputs identical to P0: `true`
- Duplicate conflicts: `0`
- Missing band cells: `0`
- File read failures: `0`
- Patch extraction errors: `0`
- Selected-file index delay after rewrite: about `0.02-0.04 s`

Observed effect:

- Selected-file index delay dropped from about `2.1 s` to about `0.02-0.04 s`.
- Local total inference fell to roughly `0.7-0.8 s` on the smoke input.

## P2: GDAL Environment + In-Memory Patches

Problem:

- The platform input puts many TIFF files in one directory.
- GDAL/rasterio can spend time scanning sibling files when opening rasters.
- The baseline inference path wrote a compressed temporary NPZ and immediately reopened it.

Change:

- Enabled GDAL open/cache settings only in `P2`:
  - `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`
  - `GDAL_READDIR_LIMIT_ON_OPEN=0`
  - `GDAL_CACHEMAX=256`
  - `VSI_CACHE=TRUE`
  - `VSI_CACHE_SIZE=33554432`
- Enabled `use_in_memory_patches: true`.
- Disabled temporary patch NPZ writing with `write_patch_npz: false`.

Latest clean two-round smoke result:

- Outputs identical to P0: `true`
- P0 median total: `0.8014 s`
- P1 median total: `0.6423 s`
- P2 median total: `0.6107 s`
- P0 median patch extraction: `0.7072 s`
- P1 median patch extraction: `0.5700 s`
- P2 median patch extraction: `0.5331 s`
- P0 median selected-file index: `0.0230 s`
- P1 median selected-file index: `0.0231 s`
- P2 median selected-file index: `0.0398 s`
- P0 median raster patch reads: `0.1948 s`
- P1 median raster patch reads: `0.1400 s`
- P2 median raster patch reads: `0.1278 s`
- P0/P1 median temporary NPZ write: about `0.0073 s`
- P2 median temporary NPZ write: `0.0 s`

Interpretation:

- P2 is safe on the local smoke because output JSON and patch diagnostics are unchanged.
- The local speed gain is smaller than P1 because the smoke has only `3` unique points.
- Platform gain could be larger because there are `171` unique extracted points and thousands of TIFFs, but this must be confirmed by platform timing.

## P3: Fast Inference Inventory + P2

Problem:

- After P1/P2, the remaining local data-prep cost was mostly TIFF inventory/audit.
- The full audit creates report-style rows that are unnecessary during submission inference.

Change:

- Added `build_fast_file_index` in `preprocessing/inventory.py`.
- Added `fast_inventory` option in `build_patch_dataset` and `run_inference`.
- Created `configs/optimization/versions/P3_submission_c03_fast_inventory_gdal_inmemory.json`.
- Updated `configs/optimization/submission_c03_working.json` to use P3 behavior.

Latest clean two-round smoke result:

- Outputs identical to P0: `true`
- P0 median total: `0.8149 s`
- P1 median total: `0.6355 s`
- P2 median total: `0.6382 s`
- P3 median total: `0.5713 s`
- P0 median patch extraction: `0.7251 s`
- P1 median patch extraction: `0.5642 s`
- P2 median patch extraction: `0.5558 s`
- P3 median patch extraction: `0.4942 s`
- P3 selected-file index phase: `0.0 s` because it is integrated into fast inventory
- P3 raster patch reads: `0.1077 s`
- P3 temporary NPZ write: `0.0 s`

Current decision:

- `P3` is the current optimized pre-model pipeline candidate.
- Keep `P0`, `P1`, and `P2` configs as backups and ablation evidence.

## Clean Experiment Area

Use:

```bash
.venv/bin/python scripts/benchmark_data_prep_p1.py --rounds 2
```

The script writes to:

- `artifacts/experiments/data_prep_pipeline/<run_name>/`
- `artifacts/experiments/data_prep_pipeline/latest_summary.json`

Latest clean run:

- Run root: `artifacts/experiments/data_prep_pipeline/run_20260430_185230/`
- Summary: `artifacts/experiments/data_prep_pipeline/latest_summary.json`

Fairness controls:

- Creates a fresh timestamped experiment directory.
- Deletes only that experiment directory if it already exists.
- Creates fresh `work/` and `output/` folders for every version and every round.
- Alternates run order across rounds.

Limit:

- It does not clear the OS file cache because that requires administrator-level system actions and would affect the whole machine. For paper results, report median timings over repeated rounds.

## Paper Wording

Use conservative wording:

> We optimized the inference data-preparation stage before model compression. Local profiling showed that selected-file index construction dominated smoke-test runtime. Replacing the groupby-heavy selector with a single-pass keyed selector reduced this stage from approximately 2.1 seconds to 0.02-0.04 seconds on a platform-shaped local smoke input while preserving identical output JSON and patch diagnostics. A second deployment-oriented variant enabled GDAL directory-scan/cache options and in-memory patch transfer, further reducing local smoke total time from about 0.81 seconds to about 0.66 seconds. Platform timing is required before claiming the same proportional speedup on hidden-test infrastructure.

Updated with two-round median benchmark:

> In a two-round local benchmark using fresh output/work directories and alternating run order, P0, P1, and P2 produced identical result JSON files. Median total runtime changed from 0.801 s for P0 to 0.642 s for P1 and 0.611 s for P2 on the 21-query smoke input. Median patch-preparation time changed from 0.707 s to 0.570 s and 0.533 s respectively. These measurements are local CPU smoke diagnostics, while final claims about platform acceleration require a new platform timing run.

Updated after P3:

> We added an inference-only fast inventory path that directly constructs the selected TIFF index while preserving the same duplicate, band-filtering, and L2A-preference rules. In a two-round local CPU smoke benchmark, P3 produced identical output to the frozen C03 reference and reduced median total runtime from 0.815 s to 0.571 s. Median data-preparation time dropped from 0.725 s to 0.494 s. These numbers are measured on a 21-query, 3-point platform-shaped smoke input, so the result is evidence of local pipeline efficiency and correctness rather than a final platform-runtime claim.

Updated after P3 platform submission:

> On the hidden-test platform, C03 P3 job `jb-aitrain-155924679797274304` confirmed that the optimized path was active: `fast_inventory=true`, `use_gdal_env=true`, `use_in_memory_patches=true`, and `write_patch_npz=false`. The run reported `35.3711 s` internal total runtime. The model forward pass took only `0.4281 s`, while patch extraction took `33.4366 s`; inside that, actual raster window extraction took `33.1392 s`. Therefore P3 is useful as a documented, cleaner pre-model pipeline, but it does not materially change card consumption because the platform bottleneck is reading many TIFF windows, not selected-file indexing, NPZ writing, postprocessing, or model forward time.

Updated after P4 local patch-array test:

> P4 batches patch reads for points that share the same selected raster file. Instead of issuing one `15x15` read per point, it computes a safe union window, reads that larger window once when the over-read is bounded, and slices the individual `15x15` patches from memory. If the union window would be too large, it falls back to the exact P3 per-point reads. On the clean local smoke input, P4 matched P3 exactly for all in-memory patch arrays and masks. Median raster extraction time changed from `0.1979 s` to `0.1694 s`, total patch dataset time changed from `0.6015 s` to `0.5646 s`, and raster read calls dropped from `540` to `180`. P4 read more total pixels (`394,740` vs `121,500`) because each union window covers extra area, so platform benefit depends on whether hidden-test points are clustered enough for fewer read calls to outweigh extra pixels.
