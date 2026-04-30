# Data-Prep Optimization Notes

Date: 2026-04-30

## Research Basis

Rasterio supports windowed reads, which is already the correct pattern for our 15x15 point patches. The caveat from Rasterio's documentation is that a small window may still cause GDAL to read one or more full internal GeoTIFF blocks, so repeated small reads are dominated by file/block access rather than the number of requested pixels.

GDAL's performance configuration documents two options directly relevant to this inference package:

- `GDAL_DISABLE_READDIR_ON_OPEN`: avoids scanning large sibling directories when opening a file. The platform puts thousands of TIFFs under `region_test`, so this is relevant.
- `GDAL_READDIR_LIMIT_ON_OPEN`: controls how many sibling files GDAL scans on open.
- `GDAL_CACHEMAX`, `VSI_CACHE`, and `VSI_CACHE_SIZE`: tune block and file caching for repeated local raster access.

Sources:

- Rasterio windowed reads: <https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html>
- GDAL configuration options: <https://gdal.org/en/stable/user/configoptions.html>

## Implemented Changes

Implemented behind the working optimization config:

- `use_gdal_env: true`
- `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`
- `GDAL_READDIR_LIMIT_ON_OPEN=0`
- `GDAL_CACHEMAX=256`
- `VSI_CACHE=TRUE`
- `VSI_CACHE_SIZE=33554432`
- `use_in_memory_patches: true`
- `write_patch_npz: false`
- `compressed_patch_npz: false`

The baseline path still writes the compressed temporary NPZ and reloads it. The optimized path keeps extracted arrays in memory for immediate inference.

Implemented globally because it preserves the selected-file result:

- Replaced the pandas groupby-heavy `select_file_index` implementation with a single-pass Python selector keyed by `(region_id, start_norm, band_id)`.
- Preserved selected path ordering, candidate counts, candidate levels, and candidate paths.

Implemented as current opt-in P3 inference path:

- Added `fast_inventory: true`.
- During inference, build the selected TIFF index directly while scanning filenames.
- Keep the full audit dataframe path for training/reporting.

Implemented as current opt-in P4 inference path:

- Added `batch_raster_reads: true`.
- For points sharing the same selected raster file, compute a bounded union window, read it once, and slice each `15x15` patch in memory.
- Fall back to exact per-point reads if the union window exceeds `max_batch_union_pixels` or `max_batch_union_overread_ratio`.
- Expose `raster_read_calls`, `raster_batch_read_calls`, `raster_fallback_patch_read_calls`, and `raster_pixels_read` in the patch report.

Implemented as current opt-in P5 inference path:

- Added `block_raster_reads: true`.
- Compute each patch window's internal GeoTIFF block IDs from `src.block_shapes`.
- Read each touched block once per selected raster file when bounded by `max_block_pixels` and `max_block_overread_ratio`.
- Fall back to P4 union-window reads, then P3 exact patch reads, when full-block reads would overread too much.
- Increased the working optimized `GDAL_CACHEMAX` to `2048`.
- Expose `raster_block_read_calls` in addition to the P4 counters.

## Verification

Repro command:

```bash
.venv/bin/python scripts/benchmark_data_prep_optimization.py
```

Latest smoke comparison:

- Outputs identical: `true`
- Baseline rows: `21`
- Optimized rows: `21`
- Baseline total before file-index optimization: about `2.97 s`
- Optimized total before file-index optimization: about `2.88 s`
- Baseline total after file-index optimization: `0.8063 s`
- Optimized total after file-index optimization: `0.6614 s`
- Baseline patch extraction after file-index optimization: `0.7164 s`
- Optimized patch extraction after file-index optimization: `0.5866 s`
- File-index selection before optimization: about `2.1 s`
- File-index selection after optimization: about `0.02-0.04 s`
- Current P3 median total in clean P0-P3 benchmark: `0.5713 s`
- Current P3 median patch extraction in clean P0-P3 benchmark: `0.4942 s`
- Current P3 output JSON matches P0: `true`
- P3 platform job `jb-aitrain-155924679797274304` reported `35.3711 s` internal total runtime and `33.1392 s` actual raster window reads; the runtime config did not include `batch_raster_reads`.
- P4 clean patch-array test in the submission repo matched P3 arrays exactly and reduced local smoke raster read calls from `540` to `180`, with no missing bands, file read failures, or patch extraction errors.
- P4 port in `project_organized` direct array check matched P3 exactly for all in-memory arrays.
- Latest P0-P4 local benchmark run root: `artifacts/experiments/data_prep_pipeline/run_20260430_203858/`.
- Latest P4 median total: `0.5491 s`; P3 in same run: `0.5516 s`; P0 in same run: `0.7325 s`.
- Latest P4 raster read calls: `180`; P3 raster read calls: `540`.
- Latest P4 batch reads: `180`; fallback patch reads: `0`; pixels read: `394740`.
- Latest P0-P5 benchmark run root: `artifacts/experiments/data_prep_pipeline/run_20260430_205725/`.
- Latest P5 output JSON matches P0: `true`.
- Latest P5 median total: `0.5494 s`; P4 in same run: `0.5532 s`; P3 in same run: `0.5521 s`.
- Latest P5 adaptive local read mode fell back to P4: `raster_block_read_calls=0`, `raster_batch_read_calls=180`, `raster_read_calls=180`.
- Forced-block local diagnostic matched P3 arrays exactly but was not faster: read calls increased from `540` to `900`, requested pixels increased from `121500` to `11145600`, and raster extraction changed from `0.1417 s` to `0.1551 s`.
- Baseline duplicate conflicts: `0`
- Optimized duplicate conflicts: `0`
- Missing band cells: `0` in both
- File read failures: `0` in both
- Patch extraction errors: `0` in both
- Global valid pixel ratio: `0.9333333333` in both

## Interpretation

This is a safe deployment optimization because predictions and patch diagnostics are unchanged. The file-index rewrite is the main local win; the GDAL environment and in-memory patch path remain useful opt-in deployment improvements. The expected platform upside should be verified with one platform diagnostic submission because the local smoke has only three unique points and benefits from OS file-cache warmup.

## Next Data-Prep Targets

- Pre-model local optimization is complete for now.
- Next meaningful check is a platform timing run with P3.
- P4 grouped raster-window reads are the next hard optimization because platform P3 timing proved raster reads dominate.
